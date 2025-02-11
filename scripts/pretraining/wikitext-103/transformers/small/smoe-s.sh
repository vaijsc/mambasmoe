mkdir -p checkpoints/wikitext-103/transformers-s/smoe

args="
--data /lustre/scratch/client/vinai/users/ducna22/data/wikitext-103 \
--base_arch transformer \
--architecture sgsgsg \
--gate_name smoe \
--nlayers 3 \
--hid-sz 128 \
--inner-hid-sz 128 \
--nheads 8 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 32 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/wikitext-103/transformers-s/smoe/smoe.pt \
"

#--data /home/ubuntu/workspace/dataset/wikitext-103 \
# --data /home/anhnd81/.cache/wikitext-103
echo "Training ..."
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 --use_env train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8