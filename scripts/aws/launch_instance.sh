#!/bin/bash
# Launch SalienceFormer EC2 instance
# Run this after your quota increase is approved

set -e

# Configuration (update these if needed)
AMI_ID="ami-0c702567ccf8b120a"  # Deep Learning AMI GPU PyTorch 2.6.0 Ubuntu 22.04
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
KEY_NAME="proxyserver"
SECURITY_GROUP="sg-08137a2072e3e3fbe"
VOLUME_SIZE=100

# Parse arguments
USE_SPOT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --spot) USE_SPOT=true; shift ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --volume-size) VOLUME_SIZE="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--spot] [--instance-type TYPE] [--volume-size GB]"
            echo "  --spot           Use spot instance (60-70% cheaper)"
            echo "  --instance-type  Instance type (default: g5.xlarge)"
            echo "  --volume-size    EBS volume size in GB (default: 100)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Launching SalienceFormer instance..."
echo "  Instance type: $INSTANCE_TYPE"
echo "  Spot: $USE_SPOT"
echo "  Volume: ${VOLUME_SIZE}GB"

if [ "$USE_SPOT" = true ]; then
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=salienceformer}]' \
        --query 'Instances[0].InstanceId' \
        --output text)
else
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=salienceformer}]' \
        --query 'Instances[0].InstanceId' \
        --output text)
fi

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "============================================"
echo "Instance is running!"
echo "============================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo ""
echo "Connect with:"
echo "  ssh -i ~/.ssh/proxyserver.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Setup SalienceFormer:"
echo "  git clone <your-repo> && cd BrainLLM"
echo "  bash scripts/aws/setup.sh"
echo ""
echo "Stop when done (to save costs):"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo ""
echo "Terminate (deletes instance):"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
