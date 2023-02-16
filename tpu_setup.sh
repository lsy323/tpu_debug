TPU_NAME=hc-tpuv4-32-1
ZONE=us-central2-b

# Prepare ssh
echo "Configure ssh...."
eval `ssh-agent`
gcloud compute config-ssh


# Clone repo
echo "Cloning repo and setup environment..."
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker all --command "
    #! /bin/bash
    cd
    git clone https://github.com/honglin-chen/tpu_debug.git
    pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl
    pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
    pip3 install torch_xla[tpuvm]
"
