from utils import *
import torch.distributed as dist
import loss
import numpy as np
from config import cfg


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


def validate(config, testloader, model, writer_dict, device):
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    prototype = np.zeros([101, 512], dtype=np.float32)
    instance_num = np.zeros([101, 1], dtype=np.float32)
    intra = np.zeros([101, 1], dtype=np.float32)
    inter = np.zeros([101, 101], dtype=np.float32)
    pro = [prototype, instance_num]
    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
            images, labels, age, name = batch
            images = images.to(device)
            age = age.to(device)
            outputs, pro, intra, inter = model(images, age, pro, intra, inter)
            ages = torch.sum(outputs * torch.Tensor([i for i in range(101)]).cuda(), dim=1)
            # print('predict is ')
            # print(ages)
            # print('label is ')
            # print(age)
            loss1 = loss.L1_loss(ages, age)
            reduced_loss = reduce_tensor(loss1)
            ave_loss.update(reduced_loss.item())
    print_loss = ave_loss.average() / world_size
    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_mae', print_loss, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss


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
    local_rank = get_rank()
    world_size = get_world_size()
    prototype = np.zeros([101, 512], dtype=np.float32)
    instance_num = np.zeros([101, 1], dtype=np.float32)
    intra = np.zeros([101, 1], dtype=np.float32)
    inter = np.zeros([101, 101], dtype=np.float32)
    pro = [prototype, instance_num]
    for i_iter, batch in enumerate(trainloader):
        images, labels, age, name = batch
        images = images.to(device)
        labels = labels.to(device)
        age = age.to(device)
        model.zero_grad()
        outputs, pro, intra, inter = model(images, age, pro, intra, inter)
        # print(np.squeeze(intra))
        # print(inter[23])
        ages = torch.sum(outputs * torch.Tensor([i for i in range(101)]).cuda(), dim=1)
        loss1 = loss.kl_loss(outputs, labels)
        loss2 = loss.L1_loss(ages, age)
        # loss3 = loss.ce_loss(outputs, labels)
        total_loss = loss1 + loss2
        # print('loss in {}'.format(total_loss))
        reduced_loss = reduce_tensor(total_loss)
        # print('loss out {}'.format(reduced_loss))
        reduced_loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss
        ave_loss.update(reduced_loss.item())
        if cfg.train.optimizer != "ADAM":
            lr = adjust_learning_rate(optimizer,
                                      base_lr,
                                      num_iters,
                                      i_iter + cur_iters, )
        else:
            scheduler = warmup_scheduler(optimizer, epoch_iters)
            scheduler.step(i_iter + cur_iters)
            lr = optimizer.param_groups[0]['lr']
            # print(optimizer.param_groups[1]['lr'])

        if i_iter % config.log.print_freq == 0 and local_rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.4e}, Loss: {:.3f} klloss:{:.3f} mloss:{:.3f}'.\
                format(epoch, num_epoch, i_iter, epoch_iters, batch_time.average(), lr, print_loss, loss1, loss2)
            logging.info(msg)

            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
