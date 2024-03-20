# Build Docker Image

Note: Install nvidia-container-toolkit first.
`nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown.`

```shell
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl restart docker
```

```shell
sudo docker run -it --rm --gpus all nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 nvidia-smi
```

Build a container image with nvidia runtime.
1. Install nvidia-container-toolkit.
```shell
sudo apt-get install nvidia-container-runtime
```
2. Edit/create the /etc/docker/daemon.json with content:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```
3. Restart docker daemon:
```shell
sudo systemctl restart docker
```
4. Disable the default docker build kit during build
```shell
DOCKER_BUILDKIT=0 docker build <blah>
```

1. Build a container image.
```shell
sudo docker build --rm -t sfss_mmsi_cu117 -f ./Dockerfile --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```

2. Start a container with remote attach.
```shell
sudo ./sfss_mmsi_run.bash
```

3. Start a jupyter notebook.
```shell
sudo ./sfss_mmsi_viz.bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

4. Start a tensorbord viewer.
```shell
sudo ./sfss_mmsi_tf.bash
tensorboard --logdir tf_logs --host 0.0.0.0
```


Debug python inside docker using debugpy and VSCode [refer](https://www.youtube.com/watch?v=ywfsLKRLmf4).
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        }
    ]
}
```