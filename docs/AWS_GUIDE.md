# SalienceFormer AWS Guide

This guide covers setting up and running SalienceFormer on AWS EC2 for consistent, reproducible experiments.

## Quick Start

```bash
# 1. Launch EC2 instance (see "Launching an Instance" below)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Run setup script
curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/BrainLLM/main/scripts/aws/setup.sh | bash

# 4. Start training
source ~/activate.sh
./scripts/aws/run_training.sh
```

## Instance Recommendations

### For Development & Testing
| Instance | GPU | VRAM | vCPUs | RAM | Cost/hr | Use Case |
|----------|-----|------|-------|-----|---------|----------|
| `g5.xlarge` | A10G | 24GB | 4 | 16GB | ~$1.00 | Quick iterations |
| `g6.xlarge` | L4 | 24GB | 4 | 16GB | ~$0.80 | Best value |

### For Full Training
| Instance | GPU | VRAM | vCPUs | RAM | Cost/hr | Use Case |
|----------|-----|------|-------|-----|---------|----------|
| `g5.2xlarge` | A10G | 24GB | 8 | 32GB | ~$1.20 | Standard training |
| `p3.2xlarge` | V100 | 16GB | 8 | 61GB | ~$3.00 | Faster training |
| `p3.8xlarge` | 4x V100 | 64GB | 32 | 244GB | ~$12.00 | Multi-GPU |

### For Ablation Studies
| Instance | GPU | VRAM | vCPUs | RAM | Cost/hr | Use Case |
|----------|-----|------|-------|-----|---------|----------|
| `g5.12xlarge` | 4x A10G | 96GB | 48 | 192GB | ~$5.00 | Parallel ablations |

**Recommendation**: Start with `g5.xlarge` for development, use `g5.2xlarge` for production training.

## Launching an Instance

### Option 1: AWS Console

1. Go to EC2 → Launch Instance
2. **Name**: `salienceformer-training`
3. **AMI**: Search for "Deep Learning AMI GPU PyTorch" (Ubuntu)
4. **Instance type**: `g5.xlarge` (or see table above)
5. **Key pair**: Create or select existing
6. **Storage**: 100GB gp3 (increase for larger datasets)
7. **Security group**: Allow SSH (port 22)
8. Launch and note the public IP

### Option 2: AWS CLI

```bash
# Set your configuration
KEY_NAME="your-key-pair"
SECURITY_GROUP="sg-xxxxxxxx"  # Must allow SSH
SUBNET_ID="subnet-xxxxxxxx"   # Your VPC subnet

# Find the latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

# Launch instance
aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type g5.xlarge \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --subnet-id $SUBNET_ID \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=salienceformer-training}]' \
    --query 'Instances[0].InstanceId' \
    --output text
```

### Option 3: Spot Instances (60-70% savings)

```bash
# Request spot instance
aws ec2 request-spot-instances \
    --spot-price "0.50" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification '{
        "ImageId": "'$AMI_ID'",
        "InstanceType": "g5.xlarge",
        "KeyName": "'$KEY_NAME'",
        "SecurityGroupIds": ["'$SECURITY_GROUP'"],
        "SubnetId": "'$SUBNET_ID'"
    }'
```

**Note**: Spot instances can be interrupted. Use checkpointing (enabled by default).

## Setup

### First-Time Setup

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone and setup (replace with your repo URL)
git clone https://github.com/YOUR_USERNAME/BrainLLM.git
cd BrainLLM
bash scripts/aws/setup.sh
```

### Subsequent Sessions

```bash
ssh -i your-key.pem ubuntu@<instance-ip>
source ~/activate.sh
```

## Running Training

### Basic Training

```bash
source ~/activate.sh
./scripts/aws/run_training.sh
```

### Custom Configuration

```bash
# More epochs, larger batch size
./scripts/aws/run_training.sh \
    --epochs 5 \
    --batch-size 16 \
    --grad-accum 2

# Different dataset
./scripts/aws/run_training.sh \
    --dataset wikitext \
    --dataset-config wikitext-103-raw-v1

# Larger model
./scripts/aws/run_training.sh \
    --base-model google/gemma-7b \
    --batch-size 4 \
    --grad-accum 8
```

### Long-Running Jobs with tmux

```bash
# Start a tmux session
tmux new -s train

# Run training
source ~/activate.sh
./scripts/aws/run_training.sh --epochs 10

# Detach: Ctrl+B, then D
# Training continues even if SSH disconnects

# Reattach later
tmux attach -t train
```

## Running Evaluation

### Standard Evaluation

```bash
source ~/activate.sh
./scripts/aws/run_evaluation.sh --datasets wikitext-2
```

### Full Evaluation Suite

```bash
./scripts/aws/run_evaluation.sh \
    --datasets wikitext-2 wikitext-103 pg19 \
    --max-samples 5000
```

### Ablation Study

```bash
./scripts/aws/run_evaluation.sh \
    --ablation \
    --ablation-seeds 5 \
    --datasets wikitext-2
```

## Managing Costs

### Monitor Usage

```bash
# Check instance running time
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=salienceformer-training" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]' \
    --output table
```

### Stop When Not in Use

```bash
# Stop instance (preserves data, stops billing for compute)
aws ec2 stop-instances --instance-ids <instance-id>

# Start again later
aws ec2 start-instances --instance-ids <instance-id>
```

### Terminate When Done

```bash
# Terminate instance (deletes everything!)
aws ec2 terminate-instances --instance-ids <instance-id>
```

### Cost Estimation

| Task | Instance | Duration | Estimated Cost |
|------|----------|----------|----------------|
| WikiText-2 (1 epoch) | g5.xlarge | ~2-3 hours | $2-3 |
| WikiText-2 (full, 3 epochs) | g5.xlarge | ~6-9 hours | $6-9 |
| WikiText-103 (1 epoch) | g5.2xlarge | ~24 hours | $30 |
| Ablation (15 variants, 3 seeds) | g5.xlarge | ~48 hours | $50 |

**Tips**:
- Use spot instances for 60-70% savings on long runs
- Stop instances when not actively training
- Use smaller datasets (WikiText-2) for development
- Run ablations in parallel on multi-GPU instances

## Data Management

### Upload Data

```bash
# From your local machine
scp -i your-key.pem data.tar.gz ubuntu@<instance-ip>:~/BrainLLM/data/
```

### Download Results

```bash
# Download outputs
scp -i your-key.pem -r ubuntu@<instance-ip>:~/BrainLLM/outputs/ ./results/

# Download specific checkpoint
scp -i your-key.pem -r ubuntu@<instance-ip>:~/BrainLLM/outputs/*/checkpoint-* ./checkpoints/
```

### Using S3 for Persistence

```bash
# On EC2 instance, save to S3
aws s3 sync ~/BrainLLM/outputs/ s3://your-bucket/salienceformer/outputs/

# Download from S3
aws s3 sync s3://your-bucket/salienceformer/outputs/ ~/BrainLLM/outputs/
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
./scripts/aws/run_training.sh --batch-size 4 --grad-accum 8

# Or use gradient checkpointing (in config)
# Enable in salienceformer/config.py: gradient_checkpointing=True
```

### CUDA Errors

```bash
# Reset GPU
sudo nvidia-smi --gpu-reset

# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### SSH Connection Timeout

```bash
# Add to your local ~/.ssh/config
Host salienceformer
    HostName <instance-ip>
    User ubuntu
    IdentityFile ~/.ssh/your-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

### Disk Full

```bash
# Check disk usage
df -h

# Clean up
rm -rf ~/BrainLLM/outputs/old_runs/
rm -rf ~/.cache/huggingface/hub/  # Clears model cache (will re-download)
```

## Reproducibility Checklist

For consistent benchmarks across experiments:

- [ ] Use the same instance type (e.g., always `g5.xlarge`)
- [ ] Use the same AMI or setup script
- [ ] Set random seeds in training args
- [ ] Document CUDA and PyTorch versions
- [ ] Save configuration with each run (automatic with scripts)
- [ ] Use the same batch size and gradient accumulation
- [ ] Store outputs in S3 for long-term persistence

## Example Workflow

```bash
# 1. Launch instance
aws ec2 run-instances ... # (see above)

# 2. SSH and setup
ssh salienceformer
bash ~/BrainLLM/scripts/aws/setup.sh

# 3. Run experiment in tmux
tmux new -s exp1
source ~/activate.sh
./scripts/aws/run_training.sh --epochs 5 --output-dir ./outputs/exp1
# Ctrl+B, D to detach

# 4. Check progress later
tmux attach -t exp1
# or check output files
tail -f ~/BrainLLM/outputs/exp1/training.log

# 5. Download results
# (from local machine)
scp -r salienceformer:~/BrainLLM/outputs/exp1 ./results/

# 6. Stop instance to save costs
aws ec2 stop-instances --instance-ids <id>
```
