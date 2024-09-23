def evaluate(model, loader_val):
    tb = time.time()

    was_training = model.training
    model.eval()

    n_sum = 0
    loss0_sum = 0.0  # loss for the criterion
    
    # 4 losses for the evaluation metric
    loss4_sum = torch.zeros(4, device=device)
    w_sum = torch.zeros(4, device=device)
    slices = [slice(0, 5), slice(5, 15), slice(15, 25)]  # spinal, foraminal, subarticular

    for d in loader_val:
        x = d['x'].to(device)  # input image
        y = d['y'].to(device)  # int (batch_size, 25)
        batch_size = len(x)

        # Predict
        with torch.no_grad():
            y_pred = model(x)  # (batch_size, 3, 25)

        w = 2 ** y  # sample_weight w = (1, 2, 4) for y = 0, 1, 2 (batch_size, 25)

        loss0 = criterion(y_pred, y)

        n_sum += batch_size
        loss0_sum += loss0.item() * batch_size

        # Compute score
        # - weighted loss for spinal, foraminal, subarticular
        # - binary cross entropy for maximum spinal severe
        ce_loss = F.cross_entropy(y_pred, y, reduction='none')  # (batch_size, 25)
        for k, idx in enumerate(slices):
            w_sum[k] += w[:, idx].sum()
            loss4_sum[k] += (w[:, idx] * ce_loss[:, idx]).sum()

        # Spinal max
        y_spinal_prob = y_pred[:, :, :5].softmax(dim=1)            # (batch_size, 3,  5)
        w_max = torch.amax(w[:, :5], dim=1)                        # (batch_size, )
        y_max = torch.amax(y[:, :5] == 2, dim=1).to(torch.float)   # 0 or 1
        y_pred_max = y_spinal_prob[:, 2, :].amax(dim=1)            # max in severe (class=2)

        loss_max = F.binary_cross_entropy(y_pred_max, y_max, reduction='none')
        loss4_sum[3] += (w_max * loss_max).sum()
        w_sum[3] += w_max.sum()

    # Average over spinal, foraminal, subarticular, and any_severe_spinal
    score = (loss4_sum / w_sum).sum().item() / 4

    model.train(was_training)

    dt = time.time() - tb
    ret = {'loss': loss0_sum / n_sum,
           'score': score,
           'dt': dt}
    return ret