import torch
from torch_geometric.datasets import Planetoid
from datetime import datetime
from torch_geometric.data import DataLoader
from model import SubGAE
from utils import create_optimizer, build_args, load_configs, set_seed
import os
import pickle
from tensorboardX import SummaryWriter
import logging

def main(args):
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    writer = SummaryWriter("experiments/logs/"+dt_string)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join("experiments/logs", dt_string, "logfile.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info(args)

    set_seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    dataset_path = 'dataset/data'
    datalist = os.listdir(dataset_path)
    data_list = []
    for data in datalist:
        with open(os.path.join(dataset_path, data), 'rb') as f:
            pkl = pickle.load(f)
            data_list.append(pkl)

    print(len(data_list))


    # Create data loaders
    loader = DataLoader(data_list, batch_size=8, shuffle=True)

    
    # Create Model
    model = SubGAE(
        bipartite=args.bipartite,
        has_edge_feat=args.has_edge_feat,
        node_mask_rate=args.node_mask_rate,
        edge_mask_rate=args.edge_mask_rate,
        enc_in_channels=args.enc_in_channels,
        enc_hid_channels=args.enc_hid_channels,
        enc_out_channels=args.enc_out_channels,
        enc_layer_num=args.enc_layer_num,
        enc_dropout=args.enc_dropout,
        enc_norm=args.enc_norm,
        enc_gnn=args.enc_gnn,
        enc_act=args.enc_act,
        dec_in_channels=args.dec_in_channels,
        dec_hid_channels=args.dec_hid_channels,
        dec_out_channels=args.dec_out_channels,
        dec_layer_num=args.dec_layer_num,
        dec_dropout=args.dec_dropout,
        dec_norm=args.dec_norm,
        dec_gnn=args.dec_gnn,
        dec_act=args.dec_act,
        mlp_hid_channels=args.mlp_hid_channels,
        mlp_layer_num=args.mlp_layer_num,
        mlp_dropout=args.mlp_dropout,
        mlp_act=args.mlp_act,
        feat_loss_func=args.feat_loss,
        expert_num=args.expert_num,
        act_expert_num=args.act_expert_num,
        device=device,
    )
    model.to(device)

    optimizer = create_optimizer(opt="adam", model=model, lr=args.lr, weight_decay=args.weight_decay)
    eta_min = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=eta_min)

    # Training    
    model.train()
    for epoch in range(args.max_epoch):

        epoch_loss = 0
        epoch_feat_loss = 0
        epoch_deg_loss = 0
        epoch_logit_loss = 0
        epoch_typ_loss = 0
        epoch_weight_loss = 0
        epoch_moe_loss = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, feat_loss, deg_loss, logit_loss, typ_loss, weight_loss, moe_loss = model.train_step(
                data=data, optimizer=optimizer, grad_norm=args.grad_norm, 
                w_feat=args.w_feat, w_deg=args.w_deg, w_logit=args.w_logit, 
                w_typ=args.w_typ, w_weight=args.w_weight, w_moe=args.w_moe,
                batch_size=args.batch_size
            )
            epoch_loss += loss
            epoch_feat_loss += feat_loss
            epoch_deg_loss += deg_loss
            epoch_logit_loss += logit_loss
            epoch_typ_loss += typ_loss
            epoch_weight_loss += weight_loss
            epoch_moe_loss += moe_loss

        ntime = datetime.now()
        logger.info(f'{ntime}')
        logger.info(f'Epoch {epoch}, Loss: {epoch_loss}')
        writer.add_scalar('Loss/Total', epoch_loss, epoch)
        writer.add_scalar('Loss/Feat', epoch_feat_loss, epoch)
        writer.add_scalar('Loss/Degree', epoch_deg_loss, epoch)
        writer.add_scalar('Loss/Logit', epoch_logit_loss, epoch)
        writer.add_scalar('Loss/Typ', epoch_typ_loss, epoch)
        writer.add_scalar('Loss/Weight', epoch_weight_loss, epoch)
        writer.add_scalar('Loss/MoE', epoch_moe_loss, epoch)
        scheduler.step()

    print(model.graphMasker.mask_token)
    torch.save(model.state_dict(), args.model_path)

if __name__ == "__main__":
    args = build_args()
    args = load_configs(args, args.cfg_path)
    main(args)
    