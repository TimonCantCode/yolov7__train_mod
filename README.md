# yolov7 train_mod

## Changes made within the train.py
This is a custom version of the train.py script which is used to train your yolov7 model. 
The changes in the script, let the train run **automatically** resume itself if running out of memory.


## The following will show the part which has been modified

```ruby
    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        try:
            model.train()

            # Update image weights (optional)
            if opt.image_weights:
                # Generate indices
                if rank in [-1, 0]:
                    cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
                # Broadcast if DDP
                if rank != -1:
                    indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                    dist.broadcast(indices, 0)
                    if rank != 0:
                        dataset.indices = indices.cpu().numpy()      

            # Update mosaic border
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(4, device=device)  # mean losses
            if rank != -1:
                dataloader.sampler.set_epoch(epoch)
            pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                        loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                    else:
                        loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if rank != -1:
                        loss *= opt.world_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Print
                if rank in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                    pbar.set_description(s)

                    # Plot
                    if plots and ni < 10:
                        f = save_dir / f'train_batch{ni}.jpg'  # filename
                        Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                        # if tb_writer:
                        #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                    elif plots and ni == 10 and wandb_logger.wandb:
                        wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                    save_dir.glob('train*.jpg') if x.exists()]})

        except torch.cuda.OutOfMemoryError:
            print("Out of memory error encountered. Attempting to resume from last checkpoint...")
            torch.cuda.empty_cache()
            if ema:
                ema.update(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.ema.state_dict() if ema else None,
                'updates': ema.updates if ema else None,
                'best_fitness': best_fitness,
                'training_results': results_file.read_text() if results_file.exists() else None,
            }, wdir / 'last.pt')
            time.sleep(10)  # Wait for 10 seconds
            ckpt = torch.load(wdir / 'last.pt', map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ema:
                ema.ema.load_state_dict(ckpt['ema_state_dict'])
                ema.updates = ckpt['updates']
            epoch = ckpt['epoch']
            del ckpt
            continue  # Continue to the next epoch

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

```

## The added parts are:
```
try:
```

```
except torch.cuda.OutOfMemoryError:
            print("Out of memory error encountered. Attempting to resume from last checkpoint...")
            torch.cuda.empty_cache()
            if ema:
                ema.update(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.ema.state_dict() if ema else None,
                'updates': ema.updates if ema else None,
                'best_fitness': best_fitness,
                'training_results': results_file.read_text() if results_file.exists() else None,
            }, wdir / 'last.pt')
            time.sleep(10)  # Wait for 10 seconds
            ckpt = torch.load(wdir / 'last.pt', map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ema:
                ema.ema.load_state_dict(ckpt['ema_state_dict'])
                ema.updates = ckpt['updates']
            epoch = ckpt['epoch']
            del ckpt
            continue  # Continue to the next epoch
```
