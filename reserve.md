# Run a single user notebook server on Chameleon

This notebook describes how to run a single user Jupyter notebook server on Chameleon. This allows you to run experiments requiring bare metal access, storage, memory, GPU and compute resources on Chameleon using a Jupyter notebook interface.

## Provision the resource

### Check resource availability

This notebook will try to reserve a RTX6000 GPU backed Ubuntu-22.04 on CHI@UC - pending availability. Before you begin, you should check the host calendar at https://chi.uc.chameleoncloud.org/project/leases/calendar/host/ to see what node types are available.

### Chameleon configuration

You can change your Chameleon project name (if not using the one that is automatically configured in the JupyterHub environment) and the site on which to reserve resources (depending on availability) in the following cell.


```python
import chi, os, time
from chi import lease
from chi import server

PROJECT_NAME = os.getenv('OS_PROJECT_NAME') # change this if you need to
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)
username = os.getenv('USER') # all exp resources will have this prefix
```

    Now using CHI@UC:
    URL: https://chi.uc.chameleoncloud.org
    Location: Argonne National Laboratory, Lemont, Illinois, USA
    Support contact: help@chameleoncloud.org


If you need to change the details of the Chameleon server, e.g. use a different OS image, or a different node type depending on availability, you can do that in the following cell.

For our sequence of notebooks, we will use a single compute node with Ubuntu 22.04 with GPU resources. (Since we'd like to use GPU for these experiments, a `gpu_rtx_6000`, `gpu_a100_nvlink`, `gpu_v100`, or `gpu_a100_pcie` node type is fine - you can change the node type to whatever is available!)


```python
image_name="CC-Ubuntu22.04-CUDA"
NODE_TYPE = "gpu_rtx_6000"
```


```python
chi.set("image", image_name)
```

### Reservation

The following cell will create a reservation that begins now, and ends in 8 hours. You can modify the start and end date as needed.


```python
start+Dat
```


```python
res = []
lease.add_node_reservation(res, node_type=NODE_TYPE, count=1)
lease.add_fip_reservation(res, count=1)
start_date, end_date = lease.lease_duration(days=0, hours=2)

l = lease.create_lease(f"{username}-{NODE_TYPE}", res, start_date=start_date, end_date=end_date)
l = lease.wait_for_active(l["id"])
```

    error: not enough resources available with query {'resource_type': 'physical:host', 'resource_properties': '["==", "$node_type", "gpu_rtx_6000"]', 'hypervisor_properties': '', 'min': 1, 'max': 1, 'start_date': datetime.datetime(2024, 4, 1, 3, 44), 'end_date': datetime.datetime(2024, 4, 1, 5, 43), 'project_id': '49f47d2c64e64937840c3f7c663a37b2', 'count_range': '1-1', 'before_end': 'default', 'on_start': 'default'}



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /tmp/ipykernel_203/4011465356.py in <cell line: 7>()
          5 
          6 l = lease.create_lease(f"{username}-{NODE_TYPE}", res, start_date=start_date, end_date=end_date)
    ----> 7 l = lease.wait_for_active(l["id"])
    

    TypeError: 'NoneType' object is not subscriptable



```python
# continue here, whether using a lease created just now or one created earlier
# l = lease.get_lease(f"{username}-{NODE_TYPE}")
```

Provisioning resources
This cell provisions resources. It will take approximately 10 minutes. You can check on its status in the Chameleon web-based UI: https://chi.uc.chameleoncloud.org/project/instances/, then come back here when it is in the READY state.


```python
reservation_id = lease.get_node_reservation(l["id"])
server.create_server(
    f"{username}-{NODE_TYPE}", 
    reservation_id=reservation_id,
    image_name=image_name
)
server_id = server.get_server_id(f"{username}-{NODE_TYPE}")
server.wait_for_active(server_id)
```

Associate an IP address with this server:


```python
reserved_fip = server.associate_floating_ip(server_id)
```

And wait for it to come up:


```python
server.wait_for_tcp(reserved_fip, port=22)
```

# Install stuff

The following cells will install some basic packages on your Chameleon server.


```python
from chi import ssh

node = ssh.Remote(reserved_fip)
```


```python
node.run('sudo apt update')
node.run('sudo apt -y install python3-pip python3-dev')
node.run('sudo pip3 install --upgrade pip')
```

    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    
    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    


    Get:1 http://security.ubuntu.com/ubuntu jammy-security InRelease [110 kB]
    Hit:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
    Hit:3 http://nova.clouds.archive.ubuntu.com/ubuntu jammy InRelease
    Hit:4 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates InRelease
    Hit:5 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-backports InRelease
    Get:6 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [1303 kB]
    Get:7 http://security.ubuntu.com/ubuntu jammy-security/main amd64 c-n-f Metadata [11.4 kB]
    Get:8 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [1616 kB]
    Get:9 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 c-n-f Metadata [520 B]
    Get:10 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [852 kB]
    Get:11 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 c-n-f Metadata [16.8 kB]
    Get:12 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [37.1 kB]
    Get:13 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 c-n-f Metadata [260 B]
    Fetched 3946 kB in 1s (3524 kB/s)
    Reading package lists...
    Building dependency tree...
    Reading state information...
    18 packages can be upgraded. Run 'apt list --upgradable' to see them.


    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    


    Reading package lists...
    Building dependency tree...
    Reading state information...
    python3-dev is already the newest version (3.10.6-1~22.04).
    The following NEW packages will be installed:
      python3-pip python3-wheel
    0 upgraded, 2 newly installed, 0 to remove and 18 not upgraded.
    Need to get 1337 kB of archives.
    After this operation, 7178 kB of additional disk space will be used.
    Get:1 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/universe amd64 python3-wheel all 0.37.1-2ubuntu0.22.04.1 [32.0 kB]
    Get:2 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/universe amd64 python3-pip all 22.0.2+dfsg-1ubuntu0.4 [1305 kB]


    debconf: unable to initialize frontend: Dialog
    debconf: (Dialog frontend will not work on a dumb terminal, an emacs shell buffer, or without a controlling terminal.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 


    Fetched 1337 kB in 0s (3681 kB/s)
    Selecting previously unselected package python3-wheel.
    (Reading database ... 92162 files and directories currently installed.)
    Preparing to unpack .../python3-wheel_0.37.1-2ubuntu0.22.04.1_all.deb ...
    Unpacking python3-wheel (0.37.1-2ubuntu0.22.04.1) ...
    Selecting previously unselected package python3-pip.
    Preparing to unpack .../python3-pip_22.0.2+dfsg-1ubuntu0.4_all.deb ...
    Unpacking python3-pip (22.0.2+dfsg-1ubuntu0.4) ...
    Setting up python3-wheel (0.37.1-2ubuntu0.22.04.1) ...
    Setting up python3-pip (22.0.2+dfsg-1ubuntu0.4) ...
    Processing triggers for man-db (2.10.2-1) ...
    
    Running kernel seems to be up-to-date.
    
    The processor microcode seems to be up-to-date.
    
    No services need to be restarted.
    
    No containers need to be restarted.
    
    No user sessions are running outdated binaries.
    
    No VM guests are running outdated hypervisor (qemu) binaries on this host.
    Requirement already satisfied: pip in /usr/lib/python3/dist-packages (22.0.2)
    Collecting pip
      Using cached pip-24.0-py3-none-any.whl (2.1 MB)
    Installing collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 22.0.2
        Not uninstalling pip at /usr/lib/python3/dist-packages, outside environment /usr
        Can't uninstall 'pip'. No files were found to uninstall.


    WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv


    Successfully installed pip-24.0





    <Result cmd='sudo pip3 install --upgrade pip' exited=0>




```python
node.run('sudo apt-get remove --purge nvidia-* -y')
node.run('sudo apt -y autoremove')
node.run('sudo apt -y install ubuntu-drivers-common')
```

    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    


    Reading package lists...
    Building dependency tree...
    Reading state information...
    The following additional packages will be installed:
      python3-xkit
    Suggested packages:
      python3-aptdaemon.pkcompat
    The following NEW packages will be installed:
      python3-xkit ubuntu-drivers-common
    0 upgraded, 2 newly installed, 0 to remove and 19 not upgraded.
    Need to get 77.2 kB of archives.
    After this operation, 415 kB of additional disk space will be used.
    Get:1 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 python3-xkit all 0.5.0ubuntu5 [18.5 kB]
    Get:2 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 ubuntu-drivers-common amd64 1:0.9.6.2~0.22.04.6 [58.7 kB]


    debconf: unable to initialize frontend: Dialog
    debconf: (Dialog frontend will not work on a dumb terminal, an emacs shell buffer, or without a controlling terminal.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 


    Fetched 77.2 kB in 1s (135 kB/s)
    Selecting previously unselected package python3-xkit.
    (Reading database ... 84806 files and directories currently installed.)
    Preparing to unpack .../python3-xkit_0.5.0ubuntu5_all.deb ...
    Unpacking python3-xkit (0.5.0ubuntu5) ...
    Selecting previously unselected package ubuntu-drivers-common.
    Preparing to unpack .../ubuntu-drivers-common_1%3a0.9.6.2~0.22.04.6_amd64.deb ...
    Unpacking ubuntu-drivers-common (1:0.9.6.2~0.22.04.6) ...
    Setting up python3-xkit (0.5.0ubuntu5) ...
    Setting up ubuntu-drivers-common (1:0.9.6.2~0.22.04.6) ...
    Created symlink /etc/systemd/system/display-manager.service.wants/gpu-manager.service → /lib/systemd/system/gpu-manager.service.
    Unit /lib/systemd/system/gpu-manager.service is added as a dependency to a non-existent unit display-manager.service.
    Created symlink /etc/systemd/system/oem-config.service.wants/gpu-manager.service → /lib/systemd/system/gpu-manager.service.
    Unit /lib/systemd/system/gpu-manager.service is added as a dependency to a non-existent unit oem-config.service.
    
    Running kernel seems to be up-to-date.
    
    The processor microcode seems to be up-to-date.
    
    No services need to be restarted.
    
    No containers need to be restarted.
    
    No user sessions are running outdated binaries.
    
    No VM guests are running outdated hypervisor (qemu) binaries on this host.





    <Result cmd='sudo apt -y install ubuntu-drivers-common' exited=0>




```python
try:
    node.run('sudo reboot') # reboot and wait for it to come up
except:
    pass
server.wait_for_tcp(reserved_fip, port=22)
```


```python
node.run('wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb')
node.run('sudo dpkg -i cuda-keyring_1.0-1_all.deb')
node.run('sudo apt update')
node.run('sudo apt -y install linux-headers-$(uname -r)')
node.run('sudo apt -y install nvidia-driver-545') 
```

    --2024-03-28 23:23:32--  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    Resolving developer.download.nvidia.com (developer.download.nvidia.com)... 152.195.19.142
    Connecting to developer.download.nvidia.com (developer.download.nvidia.com)|152.195.19.142|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4328 (4.2K) [application/x-deb]
    Saving to: ‘cuda-keyring_1.0-1_all.deb.1’
    
         0K ....                                                  100%  201M=0s
    
    2024-03-28 23:23:32 (201 MB/s) - ‘cuda-keyring_1.0-1_all.deb.1’ saved [4328/4328]
    


    (Reading database ... 84860 files and directories currently installed.)
    Preparing to unpack cuda-keyring_1.0-1_all.deb ...
    Unpacking cuda-keyring (1.0-1) over (1.0-1) ...
    Setting up cuda-keyring (1.0-1) ...


    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    


    Hit:1 http://nova.clouds.archive.ubuntu.com/ubuntu jammy InRelease
    Hit:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease
    Hit:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
    Hit:4 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates InRelease
    Hit:5 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-backports InRelease
    Hit:6 http://security.ubuntu.com/ubuntu jammy-security InRelease
    Reading package lists...
    Building dependency tree...
    Reading state information...
    19 packages can be upgraded. Run 'apt list --upgradable' to see them.


    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    


    Reading package lists...
    Building dependency tree...
    Reading state information...
    linux-headers-5.15.0-101-generic is already the newest version (5.15.0-101.111).
    0 upgraded, 0 newly installed, 0 to remove and 19 not upgraded.


    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    


    Reading package lists...
    Building dependency tree...
    Reading state information...
    The following additional packages will be installed:
      libdrm-amdgpu1 libdrm-intel1 libdrm-nouveau2 libdrm-radeon1 libegl-mesa0
      libegl1 libfontenc1 libgl1 libgl1-mesa-dri libglapi-mesa libgles2 libglvnd0
      libglx-mesa0 libglx0 libice6 libllvm15 libnvidia-cfg1-545
      libnvidia-common-545 libnvidia-compute-545 libnvidia-decode-545
      libnvidia-encode-545 libnvidia-extra-545 libnvidia-fbc1-545 libnvidia-gl-545
      libopengl0 libpciaccess0 libsensors-config libsensors5 libsm6
      libwayland-client0 libx11-xcb1 libxaw7 libxcb-dri2-0 libxcb-dri3-0
      libxcb-glx0 libxcb-present0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcvt0
      libxfixes3 libxfont2 libxkbfile1 libxmu6 libxpm4 libxshmfence1 libxt6
      libxxf86vm1 nvidia-compute-utils-545 nvidia-dkms-545
      nvidia-kernel-common-545 nvidia-kernel-source-545 nvidia-utils-545
      x11-common x11-xkb-utils xserver-common xserver-xorg-core
      xserver-xorg-video-nvidia-545
    Suggested packages:
      lm-sensors xfonts-100dpi | xfonts-75dpi xfonts-scalable
    Recommended packages:
      libgl1-amber-dri nvidia-settings nvidia-prime libnvidia-compute-545:i386
      libnvidia-decode-545:i386 libnvidia-encode-545:i386 libnvidia-fbc1-545:i386
      libnvidia-gl-545:i386 xfonts-base xcvt
    The following NEW packages will be installed:
      libdrm-amdgpu1 libdrm-intel1 libdrm-nouveau2 libdrm-radeon1 libegl-mesa0
      libegl1 libfontenc1 libgl1 libgl1-mesa-dri libglapi-mesa libgles2 libglvnd0
      libglx-mesa0 libglx0 libice6 libllvm15 libnvidia-cfg1-545
      libnvidia-common-545 libnvidia-compute-545 libnvidia-decode-545
      libnvidia-encode-545 libnvidia-extra-545 libnvidia-fbc1-545 libnvidia-gl-545
      libopengl0 libpciaccess0 libsensors-config libsensors5 libsm6
      libwayland-client0 libx11-xcb1 libxaw7 libxcb-dri2-0 libxcb-dri3-0
      libxcb-glx0 libxcb-present0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcvt0
      libxfixes3 libxfont2 libxkbfile1 libxmu6 libxpm4 libxshmfence1 libxt6
      libxxf86vm1 nvidia-compute-utils-545 nvidia-dkms-545 nvidia-driver-545
      nvidia-kernel-common-545 nvidia-kernel-source-545 nvidia-utils-545
      x11-common x11-xkb-utils xserver-common xserver-xorg-core
      xserver-xorg-video-nvidia-545
    0 upgraded, 59 newly installed, 0 to remove and 19 not upgraded.
    Need to get 324 MB of archives.
    After this operation, 922 MB of additional disk space will be used.
    Get:1 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libdrm-amdgpu1 amd64 2.4.113-2~ubuntu0.22.04.1 [19.9 kB]
    Get:2 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libpciaccess0 amd64 0.16-3 [19.1 kB]
    Get:3 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libdrm-intel1 amd64 2.4.113-2~ubuntu0.22.04.1 [66.7 kB]
    Get:4 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-cfg1-545 545.23.08-0ubuntu1 [102 kB]
    Get:5 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libdrm-nouveau2 amd64 2.4.113-2~ubuntu0.22.04.1 [17.5 kB]
    Get:6 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libdrm-radeon1 amd64 2.4.113-2~ubuntu0.22.04.1 [21.6 kB]
    Get:7 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglapi-mesa amd64 23.2.1-1ubuntu3.1~22.04.2 [37.1 kB]
    Get:8 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-common-545 545.23.08-0ubuntu1 [19.4 kB]
    Get:9 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libwayland-client0 amd64 1.20.0-1ubuntu0.1 [25.9 kB]
    Get:10 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libx11-xcb1 amd64 2:1.7.5-1ubuntu0.3 [7802 B]
    Get:11 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-dri2-0 amd64 1.14-3ubuntu3 [7206 B]
    Get:12 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-dri3-0 amd64 1.14-3ubuntu3 [6968 B]
    Get:13 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-present0 amd64 1.14-3ubuntu3 [5734 B]
    Get:14 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-compute-545 545.23.08-0ubuntu1 [48.8 MB]
    Get:15 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-sync1 amd64 1.14-3ubuntu3 [9416 B]
    Get:16 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-xfixes0 amd64 1.14-3ubuntu3 [9996 B]
    Get:17 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxshmfence1 amd64 1.3-1build4 [5394 B]
    Get:18 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libegl-mesa0 amd64 23.2.1-1ubuntu3.1~22.04.2 [118 kB]
    Get:19 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libfontenc1 amd64 1:1.1.4-1build3 [14.7 kB]
    Get:20 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libllvm15 amd64 1:15.0.7-0ubuntu0.22.04.3 [25.4 MB]
    Get:21 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libsensors-config all 1:3.6.0-7ubuntu1 [5274 B]
    Get:22 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libsensors5 amd64 1:3.6.0-7ubuntu1 [26.3 kB]
    Get:23 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgl1-mesa-dri amd64 23.2.1-1ubuntu3.1~22.04.2 [8860 kB]
    Get:24 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-decode-545 545.23.08-0ubuntu1 [1711 kB]
    Get:25 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-encode-545 545.23.08-0ubuntu1 [93.2 kB]
    Get:26 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-extra-545 545.23.08-0ubuntu1 [260 kB]
    Get:27 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-fbc1-545 545.23.08-0ubuntu1 [53.0 kB]
    Get:28 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libnvidia-gl-545 545.23.08-0ubuntu1 [150 MB]
    Get:29 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-glx0 amd64 1.14-3ubuntu3 [25.9 kB]
    Get:30 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-shm0 amd64 1.14-3ubuntu3 [5780 B]
    Get:31 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxfixes3 amd64 1:6.0.0-1 [11.7 kB]
    Get:32 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxxf86vm1 amd64 1:1.1.4-1build3 [10.4 kB]
    Get:33 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglx-mesa0 amd64 23.2.1-1ubuntu3.1~22.04.2 [158 kB]
    Get:34 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 x11-common all 1:7.7+23ubuntu2 [23.4 kB]
    Get:35 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libice6 amd64 2:1.0.10-1build2 [42.6 kB]
    Get:36 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libglvnd0 amd64 1.4.0-1 [73.6 kB]
    Get:37 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libglx0 amd64 1.4.0-1 [41.0 kB]
    Get:38 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libgl1 amd64 1.4.0-1 [110 kB]
    Get:39 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libegl1 amd64 1.4.0-1 [28.6 kB]
    Get:40 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libopengl0 amd64 1.4.0-1 [36.5 kB]
    Get:41 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libgles2 amd64 1.4.0-1 [18.0 kB]
    Get:42 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libsm6 amd64 2:1.2.3-1build2 [16.7 kB]
    Get:43 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxt6 amd64 1:1.2.1-1 [177 kB]
    Get:44 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxmu6 amd64 2:1.1.3-3 [49.6 kB]
    Get:45 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libxpm4 amd64 1:3.5.12-1ubuntu0.22.04.2 [36.7 kB]
    Get:46 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxaw7 amd64 2:1.0.14-1 [191 kB]
    Get:47 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxcvt0 amd64 0.1.1-3 [5494 B]
    Get:48 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxfont2 amd64 1:2.0.5-1build1 [94.5 kB]
    Get:49 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 libxkbfile1 amd64 1:1.1.0-1build3 [71.8 kB]
    Get:50 http://nova.clouds.archive.ubuntu.com/ubuntu jammy/main amd64 x11-xkb-utils amd64 7.7+5build4 [172 kB]
    Get:51 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 xserver-common all 2:21.1.4-2ubuntu1.7~22.04.8 [28.6 kB]
    Get:52 http://nova.clouds.archive.ubuntu.com/ubuntu jammy-updates/main amd64 xserver-xorg-core amd64 2:21.1.4-2ubuntu1.7~22.04.8 [1477 kB]
    Get:53 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  nvidia-compute-utils-545 545.23.08-0ubuntu1 [194 kB]
    Get:54 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  nvidia-kernel-source-545 545.23.08-0ubuntu1 [43.4 MB]
    Get:55 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  nvidia-kernel-common-545 545.23.08-0ubuntu1 [39.6 MB]
    Get:56 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  nvidia-dkms-545 545.23.08-0ubuntu1 [33.4 kB]
    Get:57 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  nvidia-utils-545 545.23.08-0ubuntu1 [383 kB]
    Get:58 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  xserver-xorg-video-nvidia-545 545.23.08-0ubuntu1 [1524 kB]
    Get:59 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  nvidia-driver-545 545.23.08-0ubuntu1 [479 kB]


    debconf: unable to initialize frontend: Dialog
    debconf: (Dialog frontend will not work on a dumb terminal, an emacs shell buffer, or without a controlling terminal.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 


    Fetched 324 MB in 3s (111 MB/s)
    Selecting previously unselected package libdrm-amdgpu1:amd64.
    (Reading database ... 84860 files and directories currently installed.)
    Preparing to unpack .../00-libdrm-amdgpu1_2.4.113-2~ubuntu0.22.04.1_amd64.deb ...
    Unpacking libdrm-amdgpu1:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Selecting previously unselected package libpciaccess0:amd64.
    Preparing to unpack .../01-libpciaccess0_0.16-3_amd64.deb ...
    Unpacking libpciaccess0:amd64 (0.16-3) ...
    Selecting previously unselected package libdrm-intel1:amd64.
    Preparing to unpack .../02-libdrm-intel1_2.4.113-2~ubuntu0.22.04.1_amd64.deb ...
    Unpacking libdrm-intel1:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Selecting previously unselected package libdrm-nouveau2:amd64.
    Preparing to unpack .../03-libdrm-nouveau2_2.4.113-2~ubuntu0.22.04.1_amd64.deb ...
    Unpacking libdrm-nouveau2:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Selecting previously unselected package libdrm-radeon1:amd64.
    Preparing to unpack .../04-libdrm-radeon1_2.4.113-2~ubuntu0.22.04.1_amd64.deb ...
    Unpacking libdrm-radeon1:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Selecting previously unselected package libglapi-mesa:amd64.
    Preparing to unpack .../05-libglapi-mesa_23.2.1-1ubuntu3.1~22.04.2_amd64.deb ...
    Unpacking libglapi-mesa:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Selecting previously unselected package libwayland-client0:amd64.
    Preparing to unpack .../06-libwayland-client0_1.20.0-1ubuntu0.1_amd64.deb ...
    Unpacking libwayland-client0:amd64 (1.20.0-1ubuntu0.1) ...
    Selecting previously unselected package libx11-xcb1:amd64.
    Preparing to unpack .../07-libx11-xcb1_2%3a1.7.5-1ubuntu0.3_amd64.deb ...
    Unpacking libx11-xcb1:amd64 (2:1.7.5-1ubuntu0.3) ...
    Selecting previously unselected package libxcb-dri2-0:amd64.
    Preparing to unpack .../08-libxcb-dri2-0_1.14-3ubuntu3_amd64.deb ...
    Unpacking libxcb-dri2-0:amd64 (1.14-3ubuntu3) ...
    Selecting previously unselected package libxcb-dri3-0:amd64.
    Preparing to unpack .../09-libxcb-dri3-0_1.14-3ubuntu3_amd64.deb ...
    Unpacking libxcb-dri3-0:amd64 (1.14-3ubuntu3) ...
    Selecting previously unselected package libxcb-present0:amd64.
    Preparing to unpack .../10-libxcb-present0_1.14-3ubuntu3_amd64.deb ...
    Unpacking libxcb-present0:amd64 (1.14-3ubuntu3) ...
    Selecting previously unselected package libxcb-sync1:amd64.
    Preparing to unpack .../11-libxcb-sync1_1.14-3ubuntu3_amd64.deb ...
    Unpacking libxcb-sync1:amd64 (1.14-3ubuntu3) ...
    Selecting previously unselected package libxcb-xfixes0:amd64.
    Preparing to unpack .../12-libxcb-xfixes0_1.14-3ubuntu3_amd64.deb ...
    Unpacking libxcb-xfixes0:amd64 (1.14-3ubuntu3) ...
    Selecting previously unselected package libxshmfence1:amd64.
    Preparing to unpack .../13-libxshmfence1_1.3-1build4_amd64.deb ...
    Unpacking libxshmfence1:amd64 (1.3-1build4) ...
    Selecting previously unselected package libegl-mesa0:amd64.
    Preparing to unpack .../14-libegl-mesa0_23.2.1-1ubuntu3.1~22.04.2_amd64.deb ...
    Unpacking libegl-mesa0:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Selecting previously unselected package libfontenc1:amd64.
    Preparing to unpack .../15-libfontenc1_1%3a1.1.4-1build3_amd64.deb ...
    Unpacking libfontenc1:amd64 (1:1.1.4-1build3) ...
    Selecting previously unselected package libllvm15:amd64.
    Preparing to unpack .../16-libllvm15_1%3a15.0.7-0ubuntu0.22.04.3_amd64.deb ...
    Unpacking libllvm15:amd64 (1:15.0.7-0ubuntu0.22.04.3) ...
    Selecting previously unselected package libsensors-config.
    Preparing to unpack .../17-libsensors-config_1%3a3.6.0-7ubuntu1_all.deb ...
    Unpacking libsensors-config (1:3.6.0-7ubuntu1) ...
    Selecting previously unselected package libsensors5:amd64.
    Preparing to unpack .../18-libsensors5_1%3a3.6.0-7ubuntu1_amd64.deb ...
    Unpacking libsensors5:amd64 (1:3.6.0-7ubuntu1) ...
    Selecting previously unselected package libgl1-mesa-dri:amd64.
    Preparing to unpack .../19-libgl1-mesa-dri_23.2.1-1ubuntu3.1~22.04.2_amd64.deb ...
    Unpacking libgl1-mesa-dri:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Selecting previously unselected package libxcb-glx0:amd64.
    Preparing to unpack .../20-libxcb-glx0_1.14-3ubuntu3_amd64.deb ...
    Unpacking libxcb-glx0:amd64 (1.14-3ubuntu3) ...
    Selecting previously unselected package libxcb-shm0:amd64.
    Preparing to unpack .../21-libxcb-shm0_1.14-3ubuntu3_amd64.deb ...
    Unpacking libxcb-shm0:amd64 (1.14-3ubuntu3) ...
    Selecting previously unselected package libxfixes3:amd64.
    Preparing to unpack .../22-libxfixes3_1%3a6.0.0-1_amd64.deb ...
    Unpacking libxfixes3:amd64 (1:6.0.0-1) ...
    Selecting previously unselected package libxxf86vm1:amd64.
    Preparing to unpack .../23-libxxf86vm1_1%3a1.1.4-1build3_amd64.deb ...
    Unpacking libxxf86vm1:amd64 (1:1.1.4-1build3) ...
    Selecting previously unselected package libglx-mesa0:amd64.
    Preparing to unpack .../24-libglx-mesa0_23.2.1-1ubuntu3.1~22.04.2_amd64.deb ...
    Unpacking libglx-mesa0:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Selecting previously unselected package x11-common.
    Preparing to unpack .../25-x11-common_1%3a7.7+23ubuntu2_all.deb ...
    Unpacking x11-common (1:7.7+23ubuntu2) ...
    Selecting previously unselected package libice6:amd64.
    Preparing to unpack .../26-libice6_2%3a1.0.10-1build2_amd64.deb ...
    Unpacking libice6:amd64 (2:1.0.10-1build2) ...
    Selecting previously unselected package libnvidia-cfg1-545:amd64.
    Preparing to unpack .../27-libnvidia-cfg1-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking libnvidia-cfg1-545:amd64 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libnvidia-common-545.
    Preparing to unpack .../28-libnvidia-common-545_545.23.08-0ubuntu1_all.deb ...
    Unpacking libnvidia-common-545 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libnvidia-compute-545:amd64.
    Preparing to unpack .../29-libnvidia-compute-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking libnvidia-compute-545:amd64 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libnvidia-decode-545:amd64.
    Preparing to unpack .../30-libnvidia-decode-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking libnvidia-decode-545:amd64 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libnvidia-encode-545:amd64.
    Preparing to unpack .../31-libnvidia-encode-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking libnvidia-encode-545:amd64 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libnvidia-extra-545:amd64.
    Preparing to unpack .../32-libnvidia-extra-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking libnvidia-extra-545:amd64 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libglvnd0:amd64.
    Preparing to unpack .../33-libglvnd0_1.4.0-1_amd64.deb ...
    Unpacking libglvnd0:amd64 (1.4.0-1) ...
    Selecting previously unselected package libglx0:amd64.
    Preparing to unpack .../34-libglx0_1.4.0-1_amd64.deb ...
    Unpacking libglx0:amd64 (1.4.0-1) ...
    Selecting previously unselected package libgl1:amd64.
    Preparing to unpack .../35-libgl1_1.4.0-1_amd64.deb ...
    Unpacking libgl1:amd64 (1.4.0-1) ...
    Selecting previously unselected package libnvidia-fbc1-545:amd64.
    Preparing to unpack .../36-libnvidia-fbc1-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking libnvidia-fbc1-545:amd64 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libegl1:amd64.
    Preparing to unpack .../37-libegl1_1.4.0-1_amd64.deb ...
    Unpacking libegl1:amd64 (1.4.0-1) ...
    Selecting previously unselected package libopengl0:amd64.
    Preparing to unpack .../38-libopengl0_1.4.0-1_amd64.deb ...
    Unpacking libopengl0:amd64 (1.4.0-1) ...
    Selecting previously unselected package libgles2:amd64.
    Preparing to unpack .../39-libgles2_1.4.0-1_amd64.deb ...
    Unpacking libgles2:amd64 (1.4.0-1) ...
    Selecting previously unselected package libnvidia-gl-545:amd64.
    Preparing to unpack .../40-libnvidia-gl-545_545.23.08-0ubuntu1_amd64.deb ...
    dpkg-query: no packages found matching libnvidia-gl-450
    Unpacking libnvidia-gl-545:amd64 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package libsm6:amd64.
    Preparing to unpack .../41-libsm6_2%3a1.2.3-1build2_amd64.deb ...
    Unpacking libsm6:amd64 (2:1.2.3-1build2) ...
    Selecting previously unselected package libxt6:amd64.
    Preparing to unpack .../42-libxt6_1%3a1.2.1-1_amd64.deb ...
    Unpacking libxt6:amd64 (1:1.2.1-1) ...
    Selecting previously unselected package libxmu6:amd64.
    Preparing to unpack .../43-libxmu6_2%3a1.1.3-3_amd64.deb ...
    Unpacking libxmu6:amd64 (2:1.1.3-3) ...
    Selecting previously unselected package libxpm4:amd64.
    Preparing to unpack .../44-libxpm4_1%3a3.5.12-1ubuntu0.22.04.2_amd64.deb ...
    Unpacking libxpm4:amd64 (1:3.5.12-1ubuntu0.22.04.2) ...
    Selecting previously unselected package libxaw7:amd64.
    Preparing to unpack .../45-libxaw7_2%3a1.0.14-1_amd64.deb ...
    Unpacking libxaw7:amd64 (2:1.0.14-1) ...
    Selecting previously unselected package libxcvt0:amd64.
    Preparing to unpack .../46-libxcvt0_0.1.1-3_amd64.deb ...
    Unpacking libxcvt0:amd64 (0.1.1-3) ...
    Selecting previously unselected package libxfont2:amd64.
    Preparing to unpack .../47-libxfont2_1%3a2.0.5-1build1_amd64.deb ...
    Unpacking libxfont2:amd64 (1:2.0.5-1build1) ...
    Selecting previously unselected package libxkbfile1:amd64.
    Preparing to unpack .../48-libxkbfile1_1%3a1.1.0-1build3_amd64.deb ...
    Unpacking libxkbfile1:amd64 (1:1.1.0-1build3) ...
    Selecting previously unselected package nvidia-compute-utils-545.
    Preparing to unpack .../49-nvidia-compute-utils-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking nvidia-compute-utils-545 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package nvidia-kernel-source-545.
    Preparing to unpack .../50-nvidia-kernel-source-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking nvidia-kernel-source-545 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package nvidia-kernel-common-545.
    Preparing to unpack .../51-nvidia-kernel-common-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking nvidia-kernel-common-545 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package nvidia-dkms-545.
    Preparing to unpack .../52-nvidia-dkms-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking nvidia-dkms-545 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package nvidia-utils-545.
    Preparing to unpack .../53-nvidia-utils-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking nvidia-utils-545 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package x11-xkb-utils.
    Preparing to unpack .../54-x11-xkb-utils_7.7+5build4_amd64.deb ...
    Unpacking x11-xkb-utils (7.7+5build4) ...
    Selecting previously unselected package xserver-common.
    Preparing to unpack .../55-xserver-common_2%3a21.1.4-2ubuntu1.7~22.04.8_all.deb ...
    Unpacking xserver-common (2:21.1.4-2ubuntu1.7~22.04.8) ...
    Selecting previously unselected package xserver-xorg-core.
    Preparing to unpack .../56-xserver-xorg-core_2%3a21.1.4-2ubuntu1.7~22.04.8_amd64.deb ...
    Unpacking xserver-xorg-core (2:21.1.4-2ubuntu1.7~22.04.8) ...
    Selecting previously unselected package xserver-xorg-video-nvidia-545.
    Preparing to unpack .../57-xserver-xorg-video-nvidia-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking xserver-xorg-video-nvidia-545 (545.23.08-0ubuntu1) ...
    Selecting previously unselected package nvidia-driver-545.
    Preparing to unpack .../58-nvidia-driver-545_545.23.08-0ubuntu1_amd64.deb ...
    Unpacking nvidia-driver-545 (545.23.08-0ubuntu1) ...
    Setting up libxcb-dri3-0:amd64 (1.14-3ubuntu3) ...
    Setting up libx11-xcb1:amd64 (2:1.7.5-1ubuntu0.3) ...
    Setting up libpciaccess0:amd64 (0.16-3) ...
    Setting up libdrm-nouveau2:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Setting up libxcb-xfixes0:amd64 (1.14-3ubuntu3) ...
    Setting up libxpm4:amd64 (1:3.5.12-1ubuntu0.22.04.2) ...
    Setting up libdrm-radeon1:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Setting up libglvnd0:amd64 (1.4.0-1) ...
    Setting up libxcb-glx0:amd64 (1.14-3ubuntu3) ...
    Setting up libdrm-intel1:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Setting up x11-common (1:7.7+23ubuntu2) ...
    debconf: unable to initialize frontend: Dialog
    debconf: (Dialog frontend will not work on a dumb terminal, an emacs shell buffer, or without a controlling terminal.)
    debconf: falling back to frontend: Readline
    Setting up libsensors-config (1:3.6.0-7ubuntu1) ...
    Setting up libnvidia-compute-545:amd64 (545.23.08-0ubuntu1) ...
    Setting up libxcb-shm0:amd64 (1.14-3ubuntu3) ...
    Setting up libnvidia-decode-545:amd64 (545.23.08-0ubuntu1) ...
    Setting up libopengl0:amd64 (1.4.0-1) ...
    Setting up libxxf86vm1:amd64 (1:1.1.4-1build3) ...
    Setting up libnvidia-extra-545:amd64 (545.23.08-0ubuntu1) ...
    Setting up libxcb-present0:amd64 (1.14-3ubuntu3) ...
    Setting up nvidia-kernel-source-545 (545.23.08-0ubuntu1) ...
    Setting up nvidia-utils-545 (545.23.08-0ubuntu1) ...
    Setting up libfontenc1:amd64 (1:1.1.4-1build3) ...
    Setting up libgles2:amd64 (1.4.0-1) ...
    Setting up libnvidia-common-545 (545.23.08-0ubuntu1) ...
    Setting up libxfixes3:amd64 (1:6.0.0-1) ...
    Setting up libxcb-sync1:amd64 (1.14-3ubuntu3) ...
    Setting up nvidia-compute-utils-545 (545.23.08-0ubuntu1) ...
    Warning: The home dir /nonexistent you specified can't be accessed: No such file or directory
    Adding system user `nvidia-persistenced' (UID 115) ...
    Adding new group `nvidia-persistenced' (GID 122) ...
    Adding new user `nvidia-persistenced' (UID 115) with group `nvidia-persistenced' ...
    Not creating home directory `/nonexistent'.
    Created symlink /etc/systemd/system/multi-user.target.wants/nvidia-persistenced.service → /lib/systemd/system/nvidia-persistenced.service.
    Setting up libsensors5:amd64 (1:3.6.0-7ubuntu1) ...
    Setting up libglapi-mesa:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Setting up libnvidia-cfg1-545:amd64 (545.23.08-0ubuntu1) ...
    Setting up libxcb-dri2-0:amd64 (1.14-3ubuntu3) ...
    Setting up libxshmfence1:amd64 (1.3-1build4) ...
    Setting up libllvm15:amd64 (1:15.0.7-0ubuntu0.22.04.3) ...
    Setting up libxcvt0:amd64 (0.1.1-3) ...
    Setting up libnvidia-encode-545:amd64 (545.23.08-0ubuntu1) ...
    Setting up libxkbfile1:amd64 (1:1.1.0-1build3) ...
    Setting up nvidia-kernel-common-545 (545.23.08-0ubuntu1) ...
    update-initramfs: deferring update (trigger activated)
    Setting up libxfont2:amd64 (1:2.0.5-1build1) ...
    Setting up libdrm-amdgpu1:amd64 (2.4.113-2~ubuntu0.22.04.1) ...
    Setting up libwayland-client0:amd64 (1.20.0-1ubuntu0.1) ...
    Setting up libice6:amd64 (2:1.0.10-1build2) ...
    Setting up libgl1-mesa-dri:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Setting up nvidia-dkms-545 (545.23.08-0ubuntu1) ...
    update-initramfs: deferring update (trigger activated)
    
    A modprobe blacklist file has been created at /etc/modprobe.d to prevent Nouveau
    from loading. This can be reverted by deleting the following file:
    /etc/modprobe.d/nvidia-graphics-drivers.conf
    
    A new initrd image has also been created. To revert, please regenerate your
    initrd by running the following command after deleting the modprobe.d file:
    `/usr/sbin/initramfs -u`
    
    *****************************************************************************
    *** Reboot your computer and verify that the NVIDIA graphics driver can   ***
    *** be loaded.                                                            ***
    *****************************************************************************
    
    INFO:Enable nvidia
    DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/lenovo_thinkpad
    DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/put_your_quirks_here
    DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/dell_latitude
    debconf: unable to initialize frontend: Dialog
    debconf: (Dialog frontend will not work on a dumb terminal, an emacs shell buffer, or without a controlling terminal.)
    debconf: falling back to frontend: Readline
    Loading new nvidia-545.23.08 DKMS files...
    Building for 5.15.0-101-generic
    Building for architecture x86_64
    Building initial module for 5.15.0-101-generic
    Done.
    
    nvidia.ko:
    Running module version sanity check.
     - Original module
       - No original module exists within this kernel
     - Installation
       - Installing to /lib/modules/5.15.0-101-generic/updates/dkms/
    
    nvidia-modeset.ko:
    Running module version sanity check.
     - Original module
       - No original module exists within this kernel
     - Installation
       - Installing to /lib/modules/5.15.0-101-generic/updates/dkms/
    
    nvidia-drm.ko:
    Running module version sanity check.
     - Original module
       - No original module exists within this kernel
     - Installation
       - Installing to /lib/modules/5.15.0-101-generic/updates/dkms/
    
    nvidia-peermem.ko:
    Running module version sanity check.
     - Original module
       - No original module exists within this kernel
     - Installation
       - Installing to /lib/modules/5.15.0-101-generic/updates/dkms/
    
    nvidia-uvm.ko:
    Running module version sanity check.
     - Original module
       - No original module exists within this kernel
     - Installation
       - Installing to /lib/modules/5.15.0-101-generic/updates/dkms/
    
    depmod....
    Setting up libegl-mesa0:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Setting up libegl1:amd64 (1.4.0-1) ...
    Setting up libsm6:amd64 (2:1.2.3-1build2) ...
    Setting up libglx-mesa0:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
    Setting up libglx0:amd64 (1.4.0-1) ...
    Setting up libgl1:amd64 (1.4.0-1) ...
    Setting up libnvidia-gl-545:amd64 (545.23.08-0ubuntu1) ...
    Setting up libnvidia-fbc1-545:amd64 (545.23.08-0ubuntu1) ...
    Setting up libxt6:amd64 (1:1.2.1-1) ...
    Setting up libxmu6:amd64 (2:1.1.3-3) ...
    Setting up libxaw7:amd64 (2:1.0.14-1) ...
    Setting up x11-xkb-utils (7.7+5build4) ...
    Setting up xserver-common (2:21.1.4-2ubuntu1.7~22.04.8) ...
    Setting up xserver-xorg-core (2:21.1.4-2ubuntu1.7~22.04.8) ...
    Setting up xserver-xorg-video-nvidia-545 (545.23.08-0ubuntu1) ...
    Setting up nvidia-driver-545 (545.23.08-0ubuntu1) ...
    Processing triggers for man-db (2.10.2-1) ...
    Processing triggers for dbus (1.12.20-2ubuntu4.1) ...
    Processing triggers for libc-bin (2.35-0ubuntu3.6) ...
    Processing triggers for initramfs-tools (0.140ubuntu13.4) ...
    update-initramfs: Generating /boot/initrd.img-5.15.0-101-generic
    
    Running kernel seems to be up-to-date.
    
    The processor microcode seems to be up-to-date.
    
    No services need to be restarted.
    
    No containers need to be restarted.
    
    No user sessions are running outdated binaries.
    
    No VM guests are running outdated hypervisor (qemu) binaries on this host.





    <Result cmd='sudo apt -y install nvidia-driver-545' exited=0>




```python
try:
    node.run('sudo reboot') # reboot and wait for it to come up
except:
    pass
server.wait_for_tcp(reserved_fip, port=22)
```


```python
node.run("echo 'PATH=\"/usr/local/cuda-12.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin\"' | sudo tee /etc/environment")
```

    PATH="/usr/local/cuda-12.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"





    <Result cmd='echo \'PATH="/usr/local/cuda-12.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"\' | sudo tee /etc/environment' exited=0>




```python
node = ssh.Remote(reserved_fip) # note: need a new SSH session to get new PATH
node.run('nvidia-smi')
node.run('nvcc --version')
```

    Thu Mar 28 23:42:29 2024       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  Quadro RTX 6000                On  | 00000000:3B:00.0 Off |                  Off |
    | 33%   21C    P8               4W / 260W |      1MiB / 24576MiB |      0%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+


    bash: line 1: nvcc: command not found



    ---------------------------------------------------------------------------

    UnexpectedExit                            Traceback (most recent call last)

    /tmp/ipykernel_514/488859665.py in <cell line: 3>()
          1 node = ssh.Remote(reserved_fip) # note: need a new SSH session to get new PATH
          2 node.run('nvidia-smi')
    ----> 3 node.run('nvcc --version')
    

    /opt/conda/lib/python3.10/site-packages/decorator.py in fun(*args, **kw)
        230             if not kwsyntax:
        231                 args, kw = fix(args, kw, sig)
    --> 232             return caller(func, *(extras + args), **kw)
        233     fun.__name__ = func.__name__
        234     fun.__doc__ = func.__doc__


    /opt/conda/lib/python3.10/site-packages/fabric/connection.py in opens(method, self, *args, **kwargs)
         21 def opens(method, self, *args, **kwargs):
         22     self.open()
    ---> 23     return method(self, *args, **kwargs)
         24 
         25 


    /opt/conda/lib/python3.10/site-packages/fabric/connection.py in run(self, command, **kwargs)
        761         .. versionadded:: 2.0
        762         """
    --> 763         return self._run(self._remote_runner(), command, **kwargs)
        764 
        765     @opens


    /opt/conda/lib/python3.10/site-packages/invoke/context.py in _run(self, runner, command, **kwargs)
        111     ) -> Optional[Result]:
        112         command = self._prefix_commands(command)
    --> 113         return runner.run(command, **kwargs)
        114 
        115     def sudo(self, command: str, **kwargs: Any) -> Optional[Result]:


    /opt/conda/lib/python3.10/site-packages/fabric/runners.py in run(self, command, **kwargs)
         81     def run(self, command, **kwargs):
         82         kwargs.setdefault("replace_env", True)
    ---> 83         return super().run(command, **kwargs)
         84 
         85     def read_proc_stdout(self, num_bytes):


    /opt/conda/lib/python3.10/site-packages/invoke/runners.py in run(self, command, **kwargs)
        393         """
        394         try:
    --> 395             return self._run_body(command, **kwargs)
        396         finally:
        397             if not (self._asynchronous or self._disowned):


    /opt/conda/lib/python3.10/site-packages/invoke/runners.py in _run_body(self, command, **kwargs)
        449             thread.start()
        450         # Wrap up or promise that we will, depending
    --> 451         return self.make_promise() if self._asynchronous else self._finish()
        452 
        453     def make_promise(self) -> "Promise":


    /opt/conda/lib/python3.10/site-packages/invoke/runners.py in _finish(self)
        516             raise CommandTimedOut(result, timeout=timeout)
        517         if not (result or self.opts["warn"]):
    --> 518             raise UnexpectedExit(result)
        519         return result
        520 


    UnexpectedExit: Encountered a bad command exit code!
    
    Command: 'nvcc --version'
    
    Exit code: 127
    
    Stdout: already printed
    
    Stderr: already printed
    



## Install Python packages


```python
node.run('python3 -m pip install --user tensorflow[and-cuda]')
node.run('python3 -m pip install --user numpy')
node.run('python3 -m pip install --user matplotlib')
node.run('python3 -m pip install --user seaborn')
node.run('python3 -m pip install --user librosa')
```

    Collecting librosa
      Downloading librosa-0.10.1-py3-none-any.whl.metadata (8.3 kB)
    Collecting audioread>=2.1.9 (from librosa)
      Downloading audioread-3.0.1-py3-none-any.whl.metadata (8.4 kB)
    Requirement already satisfied: numpy!=1.22.0,!=1.22.1,!=1.22.2,>=1.20.3 in ./.local/lib/python3.10/site-packages (from librosa) (1.26.4)
    Collecting scipy>=1.2.0 (from librosa)
      Downloading scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.4/60.4 kB 1.6 MB/s eta 0:00:00
    Collecting scikit-learn>=0.20.0 (from librosa)
      Downloading scikit_learn-1.4.1.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
    Collecting joblib>=0.14 (from librosa)
      Downloading joblib-1.3.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting decorator>=4.3.0 (from librosa)
      Downloading decorator-5.1.1-py3-none-any.whl.metadata (4.0 kB)
    Collecting numba>=0.51.0 (from librosa)
      Downloading numba-0.59.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.7 kB)
    Collecting soundfile>=0.12.1 (from librosa)
      Downloading soundfile-0.12.1-py2.py3-none-manylinux_2_31_x86_64.whl.metadata (14 kB)
    Collecting pooch>=1.0 (from librosa)
      Downloading pooch-1.8.1-py3-none-any.whl.metadata (9.5 kB)
    Collecting soxr>=0.3.2 (from librosa)
      Downloading soxr-0.3.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
    Requirement already satisfied: typing-extensions>=4.1.1 in ./.local/lib/python3.10/site-packages (from librosa) (4.10.0)
    Collecting lazy-loader>=0.1 (from librosa)
      Downloading lazy_loader-0.3-py3-none-any.whl.metadata (4.3 kB)
    Collecting msgpack>=1.0 (from librosa)
      Downloading msgpack-1.0.8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.1 kB)
    Collecting llvmlite<0.43,>=0.42.0dev0 (from numba>=0.51.0->librosa)
      Downloading llvmlite-0.42.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.8 kB)
    Collecting platformdirs>=2.5.0 (from pooch>=1.0->librosa)
      Downloading platformdirs-4.2.0-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: packaging>=20.0 in ./.local/lib/python3.10/site-packages (from pooch>=1.0->librosa) (24.0)
    Requirement already satisfied: requests>=2.19.0 in /usr/lib/python3/dist-packages (from pooch>=1.0->librosa) (2.25.1)
    Collecting threadpoolctl>=2.0.0 (from scikit-learn>=0.20.0->librosa)
      Downloading threadpoolctl-3.4.0-py3-none-any.whl.metadata (13 kB)
    Collecting cffi>=1.0 (from soundfile>=0.12.1->librosa)
      Downloading cffi-1.16.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting pycparser (from cffi>=1.0->soundfile>=0.12.1->librosa)
      Downloading pycparser-2.21-py2.py3-none-any.whl.metadata (1.1 kB)
    Downloading librosa-0.10.1-py3-none-any.whl (253 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253.7/253.7 kB 6.3 MB/s eta 0:00:00
    Downloading audioread-3.0.1-py3-none-any.whl (23 kB)
    Downloading decorator-5.1.1-py3-none-any.whl (9.1 kB)
    Downloading joblib-1.3.2-py3-none-any.whl (302 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 302.2/302.2 kB 26.5 MB/s eta 0:00:00
    Downloading lazy_loader-0.3-py3-none-any.whl (9.1 kB)
    Downloading msgpack-1.0.8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (385 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 385.1/385.1 kB 36.6 MB/s eta 0:00:00
    Downloading numba-0.59.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.7 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.7/3.7 MB 48.1 MB/s eta 0:00:00
    Downloading pooch-1.8.1-py3-none-any.whl (62 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.0/63.0 kB 10.6 MB/s eta 0:00:00
    Downloading scikit_learn-1.4.1.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.1 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.1/12.1 MB 53.7 MB/s eta 0:00:00
    Downloading scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.4 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.4/38.4 MB 43.8 MB/s eta 0:00:00
    Downloading soundfile-0.12.1-py2.py3-none-manylinux_2_31_x86_64.whl (1.2 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 58.2 MB/s eta 0:00:00
    Downloading soxr-0.3.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 57.9 MB/s eta 0:00:00
    Downloading cffi-1.16.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (443 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 443.9/443.9 kB 4.0 MB/s eta 0:00:00
    Downloading llvmlite-0.42.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (43.8 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.8/43.8 MB 39.3 MB/s eta 0:00:00
    Downloading platformdirs-4.2.0-py3-none-any.whl (17 kB)
    Downloading threadpoolctl-3.4.0-py3-none-any.whl (17 kB)
    Downloading pycparser-2.21-py2.py3-none-any.whl (118 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 118.7/118.7 kB 18.0 MB/s eta 0:00:00
    Installing collected packages: threadpoolctl, soxr, scipy, pycparser, platformdirs, msgpack, llvmlite, lazy-loader, joblib, decorator, audioread, scikit-learn, pooch, numba, cffi, soundfile, librosa
    Successfully installed audioread-3.0.1 cffi-1.16.0 decorator-5.1.1 joblib-1.3.2 lazy-loader-0.3 librosa-0.10.1 llvmlite-0.42.0 msgpack-1.0.8 numba-0.59.1 platformdirs-4.2.0 pooch-1.8.1 pycparser-2.21 scikit-learn-1.4.1.post1 scipy-1.12.0 soundfile-0.12.1 soxr-0.3.7 threadpoolctl-3.4.0





    <Result cmd='python3 -m pip install --user librosa' exited=0>



### Set up Jupyter on server

Install Jupyter:


```python
node.run('python3 -m pip install --user  jupyter-core jupyter-client jupyter -U --force-reinstall')
```

    Collecting jupyter-core
      Downloading jupyter_core-5.7.2-py3-none-any.whl.metadata (3.4 kB)
    Collecting jupyter-client
      Downloading jupyter_client-8.6.1-py3-none-any.whl.metadata (8.3 kB)
    Collecting jupyter
      Downloading jupyter-1.0.0-py2.py3-none-any.whl.metadata (995 bytes)
    Collecting platformdirs>=2.5 (from jupyter-core)
      Using cached platformdirs-4.2.0-py3-none-any.whl.metadata (11 kB)
    Collecting traitlets>=5.3 (from jupyter-core)
      Downloading traitlets-5.14.2-py3-none-any.whl.metadata (10 kB)
    Collecting python-dateutil>=2.8.2 (from jupyter-client)
      Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
    Collecting pyzmq>=23.0 (from jupyter-client)
      Downloading pyzmq-25.1.2-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (4.9 kB)
    Collecting tornado>=6.2 (from jupyter-client)
      Downloading tornado-6.4-cp38-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)
    Collecting notebook (from jupyter)
      Downloading notebook-7.1.2-py3-none-any.whl.metadata (10 kB)
    Collecting qtconsole (from jupyter)
      Downloading qtconsole-5.5.1-py3-none-any.whl.metadata (5.1 kB)
    Collecting jupyter-console (from jupyter)
      Downloading jupyter_console-6.6.3-py3-none-any.whl.metadata (5.8 kB)
    Collecting nbconvert (from jupyter)
      Downloading nbconvert-7.16.3-py3-none-any.whl.metadata (8.2 kB)
    Collecting ipykernel (from jupyter)
      Downloading ipykernel-6.29.4-py3-none-any.whl.metadata (6.3 kB)
    Collecting ipywidgets (from jupyter)
      Downloading ipywidgets-8.1.2-py3-none-any.whl.metadata (2.4 kB)
    Collecting six>=1.5 (from python-dateutil>=2.8.2->jupyter-client)
      Downloading six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
    Collecting comm>=0.1.1 (from ipykernel->jupyter)
      Downloading comm-0.2.2-py3-none-any.whl.metadata (3.7 kB)
    Collecting debugpy>=1.6.5 (from ipykernel->jupyter)
      Downloading debugpy-1.8.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.1 kB)
    Collecting ipython>=7.23.1 (from ipykernel->jupyter)
      Downloading ipython-8.22.2-py3-none-any.whl.metadata (4.8 kB)
    Collecting matplotlib-inline>=0.1 (from ipykernel->jupyter)
      Downloading matplotlib_inline-0.1.6-py3-none-any.whl.metadata (2.8 kB)
    Collecting nest-asyncio (from ipykernel->jupyter)
      Downloading nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)
    Collecting packaging (from ipykernel->jupyter)
      Using cached packaging-24.0-py3-none-any.whl.metadata (3.2 kB)
    Collecting psutil (from ipykernel->jupyter)
      Downloading psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)
    Collecting widgetsnbextension~=4.0.10 (from ipywidgets->jupyter)
      Downloading widgetsnbextension-4.0.10-py3-none-any.whl.metadata (1.6 kB)
    Collecting jupyterlab-widgets~=3.0.10 (from ipywidgets->jupyter)
      Downloading jupyterlab_widgets-3.0.10-py3-none-any.whl.metadata (4.1 kB)
    Collecting prompt-toolkit>=3.0.30 (from jupyter-console->jupyter)
      Downloading prompt_toolkit-3.0.43-py3-none-any.whl.metadata (6.5 kB)
    Collecting pygments (from jupyter-console->jupyter)
      Using cached pygments-2.17.2-py3-none-any.whl.metadata (2.6 kB)
    Collecting beautifulsoup4 (from nbconvert->jupyter)
      Downloading beautifulsoup4-4.12.3-py3-none-any.whl.metadata (3.8 kB)
    Collecting bleach!=5.0.0 (from nbconvert->jupyter)
      Downloading bleach-6.1.0-py3-none-any.whl.metadata (30 kB)
    Collecting defusedxml (from nbconvert->jupyter)
      Downloading defusedxml-0.7.1-py2.py3-none-any.whl.metadata (32 kB)
    Collecting jinja2>=3.0 (from nbconvert->jupyter)
      Downloading Jinja2-3.1.3-py3-none-any.whl.metadata (3.3 kB)
    Collecting jupyterlab-pygments (from nbconvert->jupyter)
      Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl.metadata (4.4 kB)
    Collecting markupsafe>=2.0 (from nbconvert->jupyter)
      Using cached MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
    Collecting mistune<4,>=2.0.3 (from nbconvert->jupyter)
      Downloading mistune-3.0.2-py3-none-any.whl.metadata (1.7 kB)
    Collecting nbclient>=0.5.0 (from nbconvert->jupyter)
      Downloading nbclient-0.10.0-py3-none-any.whl.metadata (7.8 kB)
    Collecting nbformat>=5.7 (from nbconvert->jupyter)
      Downloading nbformat-5.10.3-py3-none-any.whl.metadata (3.6 kB)
    Collecting pandocfilters>=1.4.1 (from nbconvert->jupyter)
      Downloading pandocfilters-1.5.1-py2.py3-none-any.whl.metadata (9.0 kB)
    Collecting tinycss2 (from nbconvert->jupyter)
      Downloading tinycss2-1.2.1-py3-none-any.whl.metadata (3.0 kB)
    Collecting jupyter-server<3,>=2.4.0 (from notebook->jupyter)
      Downloading jupyter_server-2.13.0-py3-none-any.whl.metadata (8.4 kB)
    Collecting jupyterlab-server<3,>=2.22.1 (from notebook->jupyter)
      Downloading jupyterlab_server-2.25.4-py3-none-any.whl.metadata (5.9 kB)
    Collecting jupyterlab<4.2,>=4.1.1 (from notebook->jupyter)
      Downloading jupyterlab-4.1.5-py3-none-any.whl.metadata (15 kB)
    Collecting notebook-shim<0.3,>=0.2 (from notebook->jupyter)
      Downloading notebook_shim-0.2.4-py3-none-any.whl.metadata (4.0 kB)
    Collecting qtpy>=2.4.0 (from qtconsole->jupyter)
      Downloading QtPy-2.4.1-py3-none-any.whl.metadata (12 kB)
    Collecting webencodings (from bleach!=5.0.0->nbconvert->jupyter)
      Downloading webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)
    Collecting decorator (from ipython>=7.23.1->ipykernel->jupyter)
      Using cached decorator-5.1.1-py3-none-any.whl.metadata (4.0 kB)
    Collecting jedi>=0.16 (from ipython>=7.23.1->ipykernel->jupyter)
      Downloading jedi-0.19.1-py2.py3-none-any.whl.metadata (22 kB)
    Collecting stack-data (from ipython>=7.23.1->ipykernel->jupyter)
      Downloading stack_data-0.6.3-py3-none-any.whl.metadata (18 kB)
    Collecting exceptiongroup (from ipython>=7.23.1->ipykernel->jupyter)
      Downloading exceptiongroup-1.2.0-py3-none-any.whl.metadata (6.6 kB)
    Collecting pexpect>4.3 (from ipython>=7.23.1->ipykernel->jupyter)
      Downloading pexpect-4.9.0-py2.py3-none-any.whl.metadata (2.5 kB)
    Collecting anyio>=3.1.0 (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading anyio-4.3.0-py3-none-any.whl.metadata (4.6 kB)
    Collecting argon2-cffi (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading argon2_cffi-23.1.0-py3-none-any.whl.metadata (5.2 kB)
    Collecting jupyter-events>=0.9.0 (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading jupyter_events-0.10.0-py3-none-any.whl.metadata (5.9 kB)
    Collecting jupyter-server-terminals (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading jupyter_server_terminals-0.5.3-py3-none-any.whl.metadata (5.6 kB)
    Collecting overrides (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading overrides-7.7.0-py3-none-any.whl.metadata (5.8 kB)
    Collecting prometheus-client (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading prometheus_client-0.20.0-py3-none-any.whl.metadata (1.8 kB)
    Collecting send2trash>=1.8.2 (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading Send2Trash-1.8.2-py3-none-any.whl.metadata (4.0 kB)
    Collecting terminado>=0.8.3 (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading terminado-0.18.1-py3-none-any.whl.metadata (5.8 kB)
    Collecting websocket-client (from jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading websocket_client-1.7.0-py3-none-any.whl.metadata (7.9 kB)
    Collecting async-lru>=1.0.0 (from jupyterlab<4.2,>=4.1.1->notebook->jupyter)
      Downloading async_lru-2.0.4-py3-none-any.whl.metadata (4.5 kB)
    Collecting httpx>=0.25.0 (from jupyterlab<4.2,>=4.1.1->notebook->jupyter)
      Downloading httpx-0.27.0-py3-none-any.whl.metadata (7.2 kB)
    Collecting jupyter-lsp>=2.0.0 (from jupyterlab<4.2,>=4.1.1->notebook->jupyter)
      Downloading jupyter_lsp-2.2.4-py3-none-any.whl.metadata (1.8 kB)
    Collecting tomli (from jupyterlab<4.2,>=4.1.1->notebook->jupyter)
      Downloading tomli-2.0.1-py3-none-any.whl.metadata (8.9 kB)
    Collecting babel>=2.10 (from jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading Babel-2.14.0-py3-none-any.whl.metadata (1.6 kB)
    Collecting json5>=0.9.0 (from jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading json5-0.9.24-py3-none-any.whl.metadata (30 kB)
    Collecting jsonschema>=4.18.0 (from jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading jsonschema-4.21.1-py3-none-any.whl.metadata (7.8 kB)
    Collecting requests>=2.31 (from jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
    Collecting fastjsonschema (from nbformat>=5.7->nbconvert->jupyter)
      Downloading fastjsonschema-2.19.1-py3-none-any.whl.metadata (2.1 kB)
    Collecting wcwidth (from prompt-toolkit>=3.0.30->jupyter-console->jupyter)
      Downloading wcwidth-0.2.13-py2.py3-none-any.whl.metadata (14 kB)
    Collecting soupsieve>1.2 (from beautifulsoup4->nbconvert->jupyter)
      Downloading soupsieve-2.5-py3-none-any.whl.metadata (4.7 kB)
    Collecting idna>=2.8 (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading idna-3.6-py3-none-any.whl.metadata (9.9 kB)
    Collecting sniffio>=1.1 (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
    Collecting typing-extensions>=4.1 (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Using cached typing_extensions-4.10.0-py3-none-any.whl.metadata (3.0 kB)
    Collecting certifi (from httpx>=0.25.0->jupyterlab<4.2,>=4.1.1->notebook->jupyter)
      Downloading certifi-2024.2.2-py3-none-any.whl.metadata (2.2 kB)
    Collecting httpcore==1.* (from httpx>=0.25.0->jupyterlab<4.2,>=4.1.1->notebook->jupyter)
      Downloading httpcore-1.0.5-py3-none-any.whl.metadata (20 kB)
    Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx>=0.25.0->jupyterlab<4.2,>=4.1.1->notebook->jupyter)
      Downloading h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)
    Collecting parso<0.9.0,>=0.8.3 (from jedi>=0.16->ipython>=7.23.1->ipykernel->jupyter)
      Downloading parso-0.8.3-py2.py3-none-any.whl.metadata (7.5 kB)
    Collecting attrs>=22.2.0 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading attrs-23.2.0-py3-none-any.whl.metadata (9.5 kB)
    Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading jsonschema_specifications-2023.12.1-py3-none-any.whl.metadata (3.0 kB)
    Collecting referencing>=0.28.4 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading referencing-0.34.0-py3-none-any.whl.metadata (2.8 kB)
    Collecting rpds-py>=0.7.1 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading rpds_py-0.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
    Collecting python-json-logger>=2.0.4 (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading python_json_logger-2.0.7-py3-none-any.whl.metadata (6.5 kB)
    Collecting pyyaml>=5.3 (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
    Collecting rfc3339-validator (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading rfc3339_validator-0.1.4-py2.py3-none-any.whl.metadata (1.5 kB)
    Collecting rfc3986-validator>=0.1.1 (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading rfc3986_validator-0.1.1-py2.py3-none-any.whl.metadata (1.7 kB)
    Collecting ptyprocess>=0.5 (from pexpect>4.3->ipython>=7.23.1->ipykernel->jupyter)
      Downloading ptyprocess-0.7.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting charset-normalizer<4,>=2 (from requests>=2.31->jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
    Collecting urllib3<3,>=1.21.1 (from requests>=2.31->jupyterlab-server<3,>=2.22.1->notebook->jupyter)
      Downloading urllib3-2.2.1-py3-none-any.whl.metadata (6.4 kB)
    Collecting argon2-cffi-bindings (from argon2-cffi->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
    Collecting executing>=1.2.0 (from stack-data->ipython>=7.23.1->ipykernel->jupyter)
      Downloading executing-2.0.1-py2.py3-none-any.whl.metadata (9.0 kB)
    Collecting asttokens>=2.1.0 (from stack-data->ipython>=7.23.1->ipykernel->jupyter)
      Downloading asttokens-2.4.1-py2.py3-none-any.whl.metadata (5.2 kB)
    Collecting pure-eval (from stack-data->ipython>=7.23.1->ipykernel->jupyter)
      Downloading pure_eval-0.2.2-py3-none-any.whl.metadata (6.2 kB)
    Collecting fqdn (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading fqdn-1.5.1-py3-none-any.whl.metadata (1.4 kB)
    Collecting isoduration (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading isoduration-20.11.0-py3-none-any.whl.metadata (5.7 kB)
    Collecting jsonpointer>1.13 (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading jsonpointer-2.4-py2.py3-none-any.whl.metadata (2.5 kB)
    Collecting uri-template (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading uri_template-1.3.0-py3-none-any.whl.metadata (8.8 kB)
    Collecting webcolors>=1.11 (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading webcolors-1.13-py3-none-any.whl.metadata (2.6 kB)
    Collecting cffi>=1.0.1 (from argon2-cffi-bindings->argon2-cffi->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Using cached cffi-1.16.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting pycparser (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Using cached pycparser-2.21-py2.py3-none-any.whl.metadata (1.1 kB)
    Collecting arrow>=0.15.0 (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading arrow-1.3.0-py3-none-any.whl.metadata (7.5 kB)
    Collecting types-python-dateutil>=2.8.10 (from arrow>=0.15.0->isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter)
      Downloading types_python_dateutil-2.9.0.20240316-py3-none-any.whl.metadata (1.8 kB)
    Downloading jupyter_core-5.7.2-py3-none-any.whl (28 kB)
    Downloading jupyter_client-8.6.1-py3-none-any.whl (105 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.9/105.9 kB 6.0 MB/s eta 0:00:00
    Downloading jupyter-1.0.0-py2.py3-none-any.whl (2.7 kB)
    Using cached platformdirs-4.2.0-py3-none-any.whl (17 kB)
    Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
    Downloading pyzmq-25.1.2-cp310-cp310-manylinux_2_28_x86_64.whl (1.1 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 30.8 MB/s eta 0:00:00
    Downloading tornado-6.4-cp38-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (435 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 435.4/435.4 kB 41.4 MB/s eta 0:00:00
    Downloading traitlets-5.14.2-py3-none-any.whl (85 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.4/85.4 kB 15.3 MB/s eta 0:00:00
    Downloading ipykernel-6.29.4-py3-none-any.whl (117 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.1/117.1 kB 25.9 MB/s eta 0:00:00
    Downloading ipywidgets-8.1.2-py3-none-any.whl (139 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.4/139.4 kB 21.1 MB/s eta 0:00:00
    Downloading jupyter_console-6.6.3-py3-none-any.whl (24 kB)
    Downloading nbconvert-7.16.3-py3-none-any.whl (257 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 257.4/257.4 kB 31.0 MB/s eta 0:00:00
    Downloading notebook-7.1.2-py3-none-any.whl (5.0 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.0/5.0 MB 83.6 MB/s eta 0:00:00
    Downloading qtconsole-5.5.1-py3-none-any.whl (123 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 123.4/123.4 kB 19.1 MB/s eta 0:00:00
    Downloading bleach-6.1.0-py3-none-any.whl (162 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 162.8/162.8 kB 25.3 MB/s eta 0:00:00
    Downloading comm-0.2.2-py3-none-any.whl (7.2 kB)
    Downloading debugpy-1.8.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 63.7 MB/s eta 0:00:00
    Downloading ipython-8.22.2-py3-none-any.whl (811 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 812.0/812.0 kB 47.5 MB/s eta 0:00:00
    Downloading Jinja2-3.1.3-py3-none-any.whl (133 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.2/133.2 kB 36.9 MB/s eta 0:00:00
    Downloading jupyter_server-2.13.0-py3-none-any.whl (383 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 383.2/383.2 kB 43.2 MB/s eta 0:00:00
    Downloading jupyterlab-4.1.5-py3-none-any.whl (11.4 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.4/11.4 MB 77.2 MB/s eta 0:00:00
    Downloading jupyterlab_server-2.25.4-py3-none-any.whl (58 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.0/59.0 kB 9.6 MB/s eta 0:00:00
    Downloading jupyterlab_widgets-3.0.10-py3-none-any.whl (215 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 215.0/215.0 kB 29.6 MB/s eta 0:00:00
    Using cached MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
    Downloading matplotlib_inline-0.1.6-py3-none-any.whl (9.4 kB)
    Downloading mistune-3.0.2-py3-none-any.whl (47 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.0/48.0 kB 8.7 MB/s eta 0:00:00
    Downloading nbclient-0.10.0-py3-none-any.whl (25 kB)
    Downloading nbformat-5.10.3-py3-none-any.whl (78 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.4/78.4 kB 14.3 MB/s eta 0:00:00
    Downloading notebook_shim-0.2.4-py3-none-any.whl (13 kB)
    Using cached packaging-24.0-py3-none-any.whl (53 kB)
    Downloading pandocfilters-1.5.1-py2.py3-none-any.whl (8.7 kB)
    Downloading prompt_toolkit-3.0.43-py3-none-any.whl (386 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 386.1/386.1 kB 39.6 MB/s eta 0:00:00
    Using cached pygments-2.17.2-py3-none-any.whl (1.2 MB)
    Downloading QtPy-2.4.1-py3-none-any.whl (93 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 93.5/93.5 kB 16.8 MB/s eta 0:00:00
    Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
    Downloading widgetsnbextension-4.0.10-py3-none-any.whl (2.3 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 67.4 MB/s eta 0:00:00
    Downloading beautifulsoup4-4.12.3-py3-none-any.whl (147 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 147.9/147.9 kB 29.0 MB/s eta 0:00:00
    Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
    Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl (15 kB)
    Downloading nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)
    Downloading psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (288 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 288.2/288.2 kB 33.2 MB/s eta 0:00:00
    Downloading tinycss2-1.2.1-py3-none-any.whl (21 kB)
    Downloading anyio-4.3.0-py3-none-any.whl (85 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.6/85.6 kB 14.8 MB/s eta 0:00:00
    Downloading async_lru-2.0.4-py3-none-any.whl (6.1 kB)
    Downloading Babel-2.14.0-py3-none-any.whl (11.0 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.0/11.0 MB 102.2 MB/s eta 0:00:00
    Downloading exceptiongroup-1.2.0-py3-none-any.whl (16 kB)
    Downloading httpx-0.27.0-py3-none-any.whl (75 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75.6/75.6 kB 13.1 MB/s eta 0:00:00
    Downloading httpcore-1.0.5-py3-none-any.whl (77 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.9/77.9 kB 13.5 MB/s eta 0:00:00
    Downloading jedi-0.19.1-py2.py3-none-any.whl (1.6 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 65.2 MB/s eta 0:00:00
    Downloading json5-0.9.24-py3-none-any.whl (30 kB)
    Downloading jsonschema-4.21.1-py3-none-any.whl (85 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.5/85.5 kB 16.0 MB/s eta 0:00:00
    Downloading jupyter_events-0.10.0-py3-none-any.whl (18 kB)
    Downloading jupyter_lsp-2.2.4-py3-none-any.whl (69 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 69.1/69.1 kB 12.1 MB/s eta 0:00:00
    Downloading pexpect-4.9.0-py2.py3-none-any.whl (63 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.8/63.8 kB 11.7 MB/s eta 0:00:00
    Downloading requests-2.31.0-py3-none-any.whl (62 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.6/62.6 kB 15.2 MB/s eta 0:00:00
    Downloading Send2Trash-1.8.2-py3-none-any.whl (18 kB)
    Downloading soupsieve-2.5-py3-none-any.whl (36 kB)
    Downloading terminado-0.18.1-py3-none-any.whl (14 kB)
    Downloading webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
    Downloading argon2_cffi-23.1.0-py3-none-any.whl (15 kB)
    Using cached decorator-5.1.1-py3-none-any.whl (9.1 kB)
    Downloading fastjsonschema-2.19.1-py3-none-any.whl (23 kB)
    Downloading jupyter_server_terminals-0.5.3-py3-none-any.whl (13 kB)
    Downloading overrides-7.7.0-py3-none-any.whl (17 kB)
    Downloading prometheus_client-0.20.0-py3-none-any.whl (54 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.5/54.5 kB 10.5 MB/s eta 0:00:00
    Downloading stack_data-0.6.3-py3-none-any.whl (24 kB)
    Downloading tomli-2.0.1-py3-none-any.whl (12 kB)
    Downloading wcwidth-0.2.13-py2.py3-none-any.whl (34 kB)
    Downloading websocket_client-1.7.0-py3-none-any.whl (58 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.5/58.5 kB 11.0 MB/s eta 0:00:00
    Downloading asttokens-2.4.1-py2.py3-none-any.whl (27 kB)
    Downloading attrs-23.2.0-py3-none-any.whl (60 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.8/60.8 kB 10.8 MB/s eta 0:00:00
    Downloading certifi-2024.2.2-py3-none-any.whl (163 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 163.8/163.8 kB 24.6 MB/s eta 0:00:00
    Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 142.1/142.1 kB 21.7 MB/s eta 0:00:00
    Downloading executing-2.0.1-py2.py3-none-any.whl (24 kB)
    Downloading idna-3.6-py3-none-any.whl (61 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.6/61.6 kB 10.8 MB/s eta 0:00:00
    Downloading jsonschema_specifications-2023.12.1-py3-none-any.whl (18 kB)
    Downloading parso-0.8.3-py2.py3-none-any.whl (100 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 18.1 MB/s eta 0:00:00
    Downloading ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)
    Downloading python_json_logger-2.0.7-py3-none-any.whl (8.1 kB)
    Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (705 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 705.5/705.5 kB 48.4 MB/s eta 0:00:00
    Downloading referencing-0.34.0-py3-none-any.whl (26 kB)
    Downloading rfc3986_validator-0.1.1-py2.py3-none-any.whl (4.2 kB)
    Downloading rpds_py-0.18.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 56.3 MB/s eta 0:00:00
    Downloading sniffio-1.3.1-py3-none-any.whl (10 kB)
    Using cached typing_extensions-4.10.0-py3-none-any.whl (33 kB)
    Downloading urllib3-2.2.1-py3-none-any.whl (121 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.1/121.1 kB 19.6 MB/s eta 0:00:00
    Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (86 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.2/86.2 kB 15.4 MB/s eta 0:00:00
    Downloading pure_eval-0.2.2-py3-none-any.whl (11 kB)
    Downloading rfc3339_validator-0.1.4-py2.py3-none-any.whl (3.5 kB)
    Using cached cffi-1.16.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (443 kB)
    Downloading h11-0.14.0-py3-none-any.whl (58 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.3/58.3 kB 10.5 MB/s eta 0:00:00
    Downloading jsonpointer-2.4-py2.py3-none-any.whl (7.8 kB)
    Downloading webcolors-1.13-py3-none-any.whl (14 kB)
    Downloading fqdn-1.5.1-py3-none-any.whl (9.1 kB)
    Downloading isoduration-20.11.0-py3-none-any.whl (11 kB)
    Downloading uri_template-1.3.0-py3-none-any.whl (11 kB)
    Downloading arrow-1.3.0-py3-none-any.whl (66 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.4/66.4 kB 11.7 MB/s eta 0:00:00
    Using cached pycparser-2.21-py2.py3-none-any.whl (118 kB)
    Downloading types_python_dateutil-2.9.0.20240316-py3-none-any.whl (9.7 kB)
    Installing collected packages: webencodings, wcwidth, pure-eval, ptyprocess, fastjsonschema, widgetsnbextension, websocket-client, webcolors, urllib3, uri-template, typing-extensions, types-python-dateutil, traitlets, tornado, tomli, tinycss2, soupsieve, sniffio, six, send2trash, rpds-py, rfc3986-validator, pyzmq, pyyaml, python-json-logger, pygments, pycparser, psutil, prompt-toolkit, prometheus-client, platformdirs, pexpect, parso, pandocfilters, packaging, overrides, nest-asyncio, mistune, markupsafe, jupyterlab-widgets, jupyterlab-pygments, jsonpointer, json5, idna, h11, fqdn, executing, exceptiongroup, defusedxml, decorator, debugpy, charset-normalizer, certifi, babel, attrs, terminado, rfc3339-validator, requests, referencing, qtpy, python-dateutil, matplotlib-inline, jupyter-core, jinja2, jedi, httpcore, comm, cffi, bleach, beautifulsoup4, async-lru, asttokens, anyio, stack-data, jupyter-server-terminals, jupyter-client, jsonschema-specifications, httpx, arrow, argon2-cffi-bindings, jsonschema, isoduration, ipython, argon2-cffi, nbformat, ipywidgets, ipykernel, qtconsole, nbclient, jupyter-events, jupyter-console, nbconvert, jupyter-server, notebook-shim, jupyterlab-server, jupyter-lsp, jupyterlab, notebook, jupyter


      WARNING: The script wsdump is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.


      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.10.0
        Uninstalling typing_extensions-4.10.0:
          Successfully uninstalled typing_extensions-4.10.0


      WARNING: The script send2trash is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.


      Attempting uninstall: pygments
        Found existing installation: Pygments 2.17.2
        Uninstalling Pygments-2.17.2:
          Successfully uninstalled Pygments-2.17.2


      WARNING: The script pygmentize is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.


      Attempting uninstall: pycparser
        Found existing installation: pycparser 2.21
        Uninstalling pycparser-2.21:
          Successfully uninstalled pycparser-2.21
      Attempting uninstall: platformdirs
        Found existing installation: platformdirs 4.2.0
        Uninstalling platformdirs-4.2.0:
          Successfully uninstalled platformdirs-4.2.0
      Attempting uninstall: packaging
        Found existing installation: packaging 24.0
        Uninstalling packaging-24.0:
          Successfully uninstalled packaging-24.0
      Attempting uninstall: markupsafe
        Found existing installation: MarkupSafe 2.1.5
        Uninstalling MarkupSafe-2.1.5:
          Successfully uninstalled MarkupSafe-2.1.5


      WARNING: The script pyjson5 is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.


      Attempting uninstall: decorator
        Found existing installation: decorator 5.1.1
        Uninstalling decorator-5.1.1:
          Successfully uninstalled decorator-5.1.1


      WARNING: The script normalizer is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script pybabel is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script qtpy is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.


      Attempting uninstall: python-dateutil
        Found existing installation: python-dateutil 2.9.0.post0
        Uninstalling python-dateutil-2.9.0.post0:
          Successfully uninstalled python-dateutil-2.9.0.post0


      WARNING: The scripts jupyter, jupyter-migrate and jupyter-troubleshoot are installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.


      Attempting uninstall: cffi
        Found existing installation: cffi 1.16.0
        Uninstalling cffi-1.16.0:
          Successfully uninstalled cffi-1.16.0


      WARNING: The scripts jupyter-kernel, jupyter-kernelspec and jupyter-run are installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script httpx is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script jsonschema is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The scripts ipython and ipython3 are installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script jupyter-trust is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script jupyter-execute is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script jupyter-events is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script jupyter-console is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The scripts jupyter-dejavu and jupyter-nbconvert are installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script jupyter-server is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The scripts jlpm, jupyter-lab, jupyter-labextension and jupyter-labhub are installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
      WARNING: The script jupyter-notebook is installed in '/home/cc/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.


    Successfully installed anyio-4.3.0 argon2-cffi-23.1.0 argon2-cffi-bindings-21.2.0 arrow-1.3.0 asttokens-2.4.1 async-lru-2.0.4 attrs-23.2.0 babel-2.14.0 beautifulsoup4-4.12.3 bleach-6.1.0 certifi-2024.2.2 cffi-1.16.0 charset-normalizer-3.3.2 comm-0.2.2 debugpy-1.8.1 decorator-5.1.1 defusedxml-0.7.1 exceptiongroup-1.2.0 executing-2.0.1 fastjsonschema-2.19.1 fqdn-1.5.1 h11-0.14.0 httpcore-1.0.5 httpx-0.27.0 idna-3.6 ipykernel-6.29.4 ipython-8.22.2 ipywidgets-8.1.2 isoduration-20.11.0 jedi-0.19.1 jinja2-3.1.3 json5-0.9.24 jsonpointer-2.4 jsonschema-4.21.1 jsonschema-specifications-2023.12.1 jupyter-1.0.0 jupyter-client-8.6.1 jupyter-console-6.6.3 jupyter-core-5.7.2 jupyter-events-0.10.0 jupyter-lsp-2.2.4 jupyter-server-2.13.0 jupyter-server-terminals-0.5.3 jupyterlab-4.1.5 jupyterlab-pygments-0.3.0 jupyterlab-server-2.25.4 jupyterlab-widgets-3.0.10 markupsafe-2.1.5 matplotlib-inline-0.1.6 mistune-3.0.2 nbclient-0.10.0 nbconvert-7.16.3 nbformat-5.10.3 nest-asyncio-1.6.0 notebook-7.1.2 notebook-shim-0.2.4 overrides-7.7.0 packaging-24.0 pandocfilters-1.5.1 parso-0.8.3 pexpect-4.9.0 platformdirs-4.2.0 prometheus-client-0.20.0 prompt-toolkit-3.0.43 psutil-5.9.8 ptyprocess-0.7.0 pure-eval-0.2.2 pycparser-2.21 pygments-2.17.2 python-dateutil-2.9.0.post0 python-json-logger-2.0.7 pyyaml-6.0.1 pyzmq-25.1.2 qtconsole-5.5.1 qtpy-2.4.1 referencing-0.34.0 requests-2.31.0 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 rpds-py-0.18.0 send2trash-1.8.2 six-1.16.0 sniffio-1.3.1 soupsieve-2.5 stack-data-0.6.3 terminado-0.18.1 tinycss2-1.2.1 tomli-2.0.1 tornado-6.4 traitlets-5.14.2 types-python-dateutil-2.9.0.20240316 typing-extensions-4.10.0 uri-template-1.3.0 urllib3-2.2.1 wcwidth-0.2.13 webcolors-1.13 webencodings-0.5.1 websocket-client-1.7.0 widgetsnbextension-4.0.10





    <Result cmd='python3 -m pip install --user  jupyter-core jupyter-client jupyter -U --force-reinstall' exited=0>



### Run a JupyterHub server

Run the following cell


```python
print('ssh -L 127.0.0.1:8888:127.0.0.1:8888 cc@' + reserved_fip) 
```

    ssh -L 127.0.0.1:8888:127.0.0.1:8888 cc@192.5.86.241


then paste its output into a local terminal on your own device, to set up a tunnel to the Jupyter server. If your Chameleon key is not in the default location, you should also specify the path to your key as an argument, using -i. Leave this SSH session open.

Then, run the following cell, which will start a command that does not terminate:


```python
node.run("/home/cc/.local/bin/jupyter notebook --port=8888 --notebook-dir='/home/cc/ml-energy/notebooks'")
```

    [I 2024-03-29 00:20:55.773 ServerApp] jupyter_lsp | extension was successfully linked.
    [I 2024-03-29 00:20:55.777 ServerApp] jupyter_server_terminals | extension was successfully linked.
    [I 2024-03-29 00:20:55.780 ServerApp] jupyterlab | extension was successfully linked.
    [I 2024-03-29 00:20:55.783 ServerApp] notebook | extension was successfully linked.
    [I 2024-03-29 00:20:55.784 ServerApp] Writing Jupyter server cookie secret to /home/cc/.local/share/jupyter/runtime/jupyter_cookie_secret
    [I 2024-03-29 00:20:55.956 ServerApp] notebook_shim | extension was successfully linked.
    [I 2024-03-29 00:20:55.969 ServerApp] notebook_shim | extension was successfully loaded.
    [I 2024-03-29 00:20:55.971 ServerApp] jupyter_lsp | extension was successfully loaded.
    [I 2024-03-29 00:20:55.972 ServerApp] jupyter_server_terminals | extension was successfully loaded.
    [I 2024-03-29 00:20:55.973 LabApp] JupyterLab extension loaded from /home/cc/.local/lib/python3.10/site-packages/jupyterlab
    [I 2024-03-29 00:20:55.973 LabApp] JupyterLab application directory is /home/cc/.local/share/jupyter/lab
    [I 2024-03-29 00:20:55.973 LabApp] Extension Manager is 'pypi'.
    [I 2024-03-29 00:20:56.016 ServerApp] jupyterlab | extension was successfully loaded.
    [I 2024-03-29 00:20:56.018 ServerApp] notebook | extension was successfully loaded.
    [I 2024-03-29 00:20:56.019 ServerApp] Serving notebooks from local directory: /home/cc/ml-energy/notebooks
    [I 2024-03-29 00:20:56.019 ServerApp] Jupyter Server 2.13.0 is running at:
    [I 2024-03-29 00:20:56.019 ServerApp] http://localhost:8888/tree?token=1d7da234ccb16bca60b7f7269cc513369235c9d11b4013a6
    [I 2024-03-29 00:20:56.019 ServerApp]     http://127.0.0.1:8888/tree?token=1d7da234ccb16bca60b7f7269cc513369235c9d11b4013a6
    [I 2024-03-29 00:20:56.019 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [W 2024-03-29 00:20:56.022 ServerApp] No web browser found: Error('could not locate runnable browser').
    [C 2024-03-29 00:20:56.022 ServerApp] 
        
        To access the server, open this file in a browser:
            file:///home/cc/.local/share/jupyter/runtime/jpserver-2194-open.html
        Or copy and paste one of these URLs:
            http://localhost:8888/tree?token=1d7da234ccb16bca60b7f7269cc513369235c9d11b4013a6
            http://127.0.0.1:8888/tree?token=1d7da234ccb16bca60b7f7269cc513369235c9d11b4013a6
    [I 2024-03-29 00:20:56.054 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server
    [I 2024-03-29 00:50:27.644 ServerApp] Creating new notebook in 
    [I 2024-03-29 00:50:27.698 ServerApp] Writing notebook-signing key to /home/cc/.local/share/jupyter/notebook_secret
    [I 2024-03-29 00:50:28.010 ServerApp] Saving file at /Untitled.ipynb
    [I 2024-03-29 00:50:28.036 ServerApp] Kernel started: 173d9618-5c14-4517-9583-f46fa0e175d8
    [I 2024-03-29 00:50:28.487 ServerApp] Connecting to kernel 173d9618-5c14-4517-9583-f46fa0e175d8.
    [I 2024-03-29 00:50:28.629 ServerApp] Connecting to kernel 173d9618-5c14-4517-9583-f46fa0e175d8.
    [I 2024-03-29 00:50:28.737 ServerApp] Connecting to kernel 173d9618-5c14-4517-9583-f46fa0e175d8.
    [I 2024-03-29 00:50:28.897 ServerApp] Connecting to kernel 173d9618-5c14-4517-9583-f46fa0e175d8.
    [W 2024-03-29 00:50:29.007 ServerApp] Got events for closed stream <zmq.eventloop.zmqstream.ZMQStream object at 0x7f63744ef070>
    [I 2024-03-29 00:50:29.284 ServerApp] Connecting to kernel 173d9618-5c14-4517-9583-f46fa0e175d8.
    2024-03-29 00:50:45.731058: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-29 00:50:46.464127: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    [I 2024-03-29 00:52:30.154 ServerApp] Saving file at /Untitled.ipynb


In the output of the cell above, look for a URL in this format:

http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
Copy this URL and open it in a browser. Then, you can run the sequence of notebooks that you'll see there, in order.

If you need to stop and re-start your Jupyter server,

* Use Kernel > Interrupt Kernel twice to stop the cell above
* Then run the following cell to kill whatever may be left running in the background.


```python
node.run("sudo killall jupyter-notebook")
```

# Release resources

If you finish with your experimentation before your lease expires,release your resources and tear down your environment by running the following (commented out to prevent accidental deletions).

This section is designed to work as a "standalone" portion - you can come back to this notebook, ignore the top part, and just run this section to delete your reasources.




```python
# setup environment - if you made any changes in the top part, make the same changes here
import chi, os
from chi import lease, server

PROJECT_NAME = os.getenv('OS_PROJECT_NAME')
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)

lease = chi.lease.get_lease(f"{os.getenv('USER')}-{NODE_TYPE}")
```

    Now using CHI@UC:
    URL: https://chi.uc.chameleoncloud.org
    Location: Argonne National Laboratory, Lemont, Illinois, USA
    Support contact: help@chameleoncloud.org



```python
# DELETE = False
DELETE = True 

if DELETE:
    # delete server
    server_id = chi.server.get_server_id(f"{os.getenv('USER')}-{NODE_TYPE}")
    chi.server.delete_server(server_id)

    # release floating IP
    reserved_fip =  chi.lease.get_reserved_floating_ips(lease["id"])[0]
    ip_info = chi.network.get_floating_ip(reserved_fip)
    chi.neutron().delete_floatingip(ip_info["id"])

    # delete lease
    chi.lease.delete_lease(lease["id"])
```

    Deleted lease with id 27275808-e9ee-4269-a436-e3c1423c726a

