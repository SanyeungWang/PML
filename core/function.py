import time
import torch
import logging
from utils import AverageMeter, get_rank, get_world_size, adjust_learning_rate
import torch.distributed as dist
import loss


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, optimizer, model, writer_dict, device):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    rank = torch.Tensor([i for i in range(101)]).cuda()
    for i_iter, batch in enumerate(trainloader):
        images, labels, age = batch
        images = images.to(device)
        labels = labels.to(device)
        age = age.to(device)
        model.zero_grad()
        outputs = model(images)
        ages = torch.sum(outputs*rank, dim=1)
        loss1 = loss.kl_loss(outputs, labels)
        loss2 = loss.L1_loss(ages, age)
        total_loss = loss1 + loss2
        reduced_loss = reduce_tensor(total_loss)
        reduced_loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss
        ave_loss.update(reduced_loss.item())
        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), lr, print_loss)
            logging.info(msg)

            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
