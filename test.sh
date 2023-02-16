TPU_NAME=hc-tpuv4-32-1
ZONE=us-central2-b

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker all --command "
    cd ~/tpu_debug;
    PJRT_DEVICE=TPU MASTER_ADDR=localhost MASTER_PORT=12500 python3 test_script.py
"