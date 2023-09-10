import argparse

dataset = 'zappos'    #zappos|cgqa
base_root =  ""
DATA_FOLDER = ""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default=base_root+dataset+'.yml')
parser.add_argument('--config_name', default='')
parser.add_argument('--base_root', default=base_root)
parser.add_argument('--dataset_name', default=dataset)
parser.add_argument('--root_dir', default=DATA_FOLDER)
parser.add_argument('--splitname', default='compositional-split-natural')
parser.add_argument('--wordembs', default='glove')
parser.add_argument('--emb_dim', default=300)
parser.add_argument('--checkpoint_dir', default=DATA_FOLDER+dataset)
parser.add_argument('--start_epoch', default=1)
parser.add_argument('--num_workers', default=2)
parser.add_argument('--test_batch_size', default=128)
parser.add_argument('--save_every_epoch', default=1)
parser.add_argument('--eval_every_epoch', default=1)
parser.add_argument('--topk', default=1)
parser.add_argument('--wd', default=0.00005)
parser.add_argument('--batch_size', default=512)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--num_negs', default=40)
