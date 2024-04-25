:::{.cell}

## Run a single user notebook server on Chameleon with space reserved on share

This notebook describes how to run a single user Jupyter notebook server on Chameleon with external space reserved on share. This allows you to run experiments requiring bare metal access, storage, memory, GPU and compute resources on Chameleon using a Jupyter notebook interface.
:::

:::{.cell}
### Provision the resource

#### Check resource availability

This notebook will try to reserve a RTX6000 GPU backed Ubuntu-22.04 on CHI@UC - pending availability. Before you begin, you should check the host calendar at https://chi.uc.chameleoncloud.org/project/leases/calendar/host/ to see what node types are available.

#### Chameleon configuration

You can change your Chameleon project name (if not using the one that is automatically configured in the JupyterHub environment) and the site on which to reserve resources (depending on availability) in the following cell.
:::

:::{.cell .code}
```
import chi, os, time
from chi import lease
from chi import server
from chi import network
from chi import share

project_name = "CHI-231095" # Replace with your project name
region_name = "CHI@UC"     # Replace site if not use CHI@UC

chi.set('project_name', project_name)
chi.set('region_name', region_name)
chi.use_site(region_name)
username = os.getenv('USER') # all exp resources will have this prefix
```
:::


:::{.cell .code}
```
image_name="CC-Ubuntu22.04-CUDA"
NODE_TYPE = "gpu_rtx_6000"


NAME = f"{username}-{NODE_TYPE}-share" # a name for all openstack components
NAME
```
:::


:::{.cell}
### Create a share

In this step, you will create a 1024 GiB share. A share is a pre-allocated storage space at a CephFS.

**Note**: There are no charges for the storage spaces of your shares. However, there is a limit on the total size and the number of shares you can create within your project. The maximum number of shares is 10 and the maximum size allowed for all shares in a project is 2000 GiB.
:::

:::{.cell .code}
```
my_share= share.create_share(size=1024, name=NAME) #Comment this line if you're using an already created share and uncomment the line below
# my_share= share.get_share(NAME)
my_share
```
:::

:::{.cell}
### Reservation

The following cell will create a reservation that begins now, and ends in 8 hours. You can modify the start and end date as needed.
:::

:::{.cell .code}
```
res = []
lease.add_node_reservation(res, node_type=NODE_TYPE, count=1)
lease.add_network_reservation(res, network_name=NAME, resource_properties=["==", "$usage_type", "storage"])
lease.add_fip_reservation(res, count=1)

start_date, end_date = chi.lease.lease_duration(days=0, hours=72)
# if you won't start right now - comment the line above, uncomment two lines below
# start_date = '2024-04-21 08:51' # manually define to desired start time 
# end_date = '2024-04-22 12:55' # manually define to desired start time 


l = lease.create_lease(NAME, res, start_date=start_date, end_date=end_date)
l = lease.wait_for_active(l["id"]) #Comment this line if the lease starts in the future
```
:::

:::{.cell .code}
```
# continue here, whether using a lease created just now or one created earlier
l = lease.get_lease(NAME)
```
:::

:::{.cell}
### Provisioning resources

This cell provisions resources. It will take approximately 10 minutes. You can check on its status in the Chameleon web-based UI: https://chi.uc.chameleoncloud.org/project/instances/, then come back here when it is in the READY state.
:::

:::{.cell .code}
```
# Create a server
reservation_id = lease.get_node_reservation(l["id"])
server_ = server.create_server(NAME, 
                                  reservation_id=reservation_id, 
                                  network_name=NAME, 
                                  image_name=image_name)


server_id = server.get_server_id(NAME)
# Wait until the server is active
server.wait_for_active(server_id)
```
:::

:::{.cell}
## Talk to the server

To attach floating IP to your instance created on a storage network, you need to create a router with `public` external network. Then connect the storage subnet to the router. You must specify an unused IP address which belongs to the selected subnet.
:::

:::{.cell .code}
```
# Get storage network id
network_ = network.get_network(NAME)
network_id = network_['id']

# Get an unused IP address on the storage subnet and create a port
subnet_id = network.get_subnet_id(NAME + '-subnet')
port = network.create_port(NAME, network_id, subnet_id=subnet_id)

# Create a router with public external network
router = network.create_router(NAME, gw_network_name='public')

# Added port to router
network.add_port_to_router_by_name(NAME, NAME)
```
:::

:::{.cell}
Associate an IP address with this server:
:::

:::{.cell .code}
```
from chi import ssh

floating_ip = server.associate_floating_ip(server_id) #Comment this line if you're going to work on an already existing server and uncomment the line below
# floating_ip = lease.get_reserved_floating_ips(l["id"])[0]
```
:::

:::{.cell}

## View share and check access rules

The paths of the export locations are important as you will use this path to mount your share to your bare metal instance. Also, the accessibility of the shares are controlled internally by the reservation service. You need to check if the access rules are granted to the share.
:::

:::{.cell .code}
```
share_ = share.get_share(my_share.id)

# Get export path
export_path = share_.export_locations[0]

# Get and check access rules
subnet = network.get_subnet(NAME + '-subnet')
access_rules = share.get_access_rules(share_.id)
access_rule_found = False
for rule in access_rules:
    print(rule)
    if rule.state == "active" and rule.access_to == subnet['cidr'] and rule.access_level == "rw":
        access_rule_found = True
        print("Access rule has been added successfully!")
        break
if not access_rule_found:
    print("Failed to find the access rule!")
```
:::

:::{.cell .code}
```
server.wait_for_tcp(floating_ip, 22)

# Create a remote connection
node = ssh.Remote(floating_ip)
```
:::

:::{.cell}

## Mount the share

Mounting your share to your instance is simple with the `mount` command.
:::

:::{.cell .code}
```
mnt_dir = "/mnt"

# Mount to mnt_dir 
node.sudo(f"mount -t nfs -o nfsvers=4.2,proto=tcp {export_path} {mnt_dir}", hide=True)

# Add a file to share
node.sudo(f"bash -c 'echo \"this is my test file\" > {mnt_dir}/mytext.txt'", hide=True)

# List mnt_dir
list_files = node.sudo(f"ls -la {mnt_dir}", hide=True)
print(list_files.stdout)

# Unmount - Uncomment this line if you would like to unmount the storage
# node.sudo(f"umount {mnt_dir}", hide=True)
```
:::

:::{.cell}

### Install Stuff

The following cells will install some basic packages for your Chameleon server.
:::

:::{.cell .code}
```
node.run('sudo apt update')
node.run('sudo apt -y install python3-pip python3-dev')
node.run('sudo pip3 install --upgrade pip')
node.run('sudo apt -y install libcudnn8=8.9.6.50-1+cuda12.2') #Installing appropriate version of cudnn for the installed drivers
node.run('sudo apt -y install pandoc')
node.run('sudo apt -y install ffmpeg')
```
:::

:::{.cell}
Add cuda to the environment path to ensure that the machine can identify the drivers
:::

:::{.cell .code}
```
node.run("echo 'PATH=\"/usr/local/cuda-12.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin\"' | sudo tee /etc/environment")
```
:::

:::{.cell}
Now reboot the server for all the installations to patch correctly
:::

:::{.cell .code}
```
try:
    node.run('sudo reboot') # reboot and wait for it to come up
except:
    pass
server.wait_for_tcp(floating_ip, port=22)
```
:::

:::{.cell .code}
```
node = ssh.Remote(floating_ip) # note: need a new SSH session to get new PATH
node.run('nvidia-smi')
node.run('nvcc --version')
```
:::

:::{.cell}
## Install Python packages
:::

:::{.cell .code}
```
node.run('python3 -m pip install --user tensorflow[and-cuda]==2.16.1')
node.run('python3 -m pip install --user numpy==1.26.4')
node.run('python3 -m pip install --user matplotlib==3.8.4')
node.run('python3 -m pip install --user seaborn==0.13.2')
node.run('python3 -m pip install --user librosa==0.10.1')
node.run('python3 -m pip install --user zeus-ml==0.8.2')
node.run('python3 -m pip install --user torch==2.2.2 torchvision==0.17.2 torchaudio==-2.2.2')
node.run('python3 -m pip install --user pydot==2.0.0')
```
:::

:::{.cell}
Test your installation - make sure Tensorflow can see the GPU:
:::


:::{.cell .code}
```
node.run('python3 -c \'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))\'')
# should say: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
:::

:::{.cell}
Now make sure torch can see the GPU:
:::

:::{.cell .code}
```
node.run('python3 -c \'import torch; print(torch.cuda.get_device_name(0))\'')
# should say: Quadro RTX 6000
```
:::

:::{.cell}

## Setup Jupyter on server

Install Jupyter
:::

:::{.cell .code}
```
node.run('python3 -m pip install --user  jupyter-core jupyter-client jupyter -U --force-reinstall')
```
:::


:::{.cell}

## Retrieve the materials

Finally, get a copy of the notebooks that you will run:
:::

:::{.cell .code}
```
node.run('git clone https://github.com/teaching-on-testbeds/ml-energy.git')
```
:::

:::{.cell}

## Run a JupyterHub server

Run the following cell
:::

:::{.cell .code}
```
print('ssh -L 127.0.0.1:8888:127.0.0.1:8888 cc@' + floating_ip) 
```
:::

:::{.cell}
then paste its output into a local terminal on your own device, to set up a tunnel to the Jupyter server. Make sure that 8888 port on your local machine is free before running this command. Upon successful login and tunneling, you should see this output.

```
Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-101-generic x86_64)
Last login: xxxxxxxxxxxxxx
```

If your Chameleon key is not in the default location, you should also specify the path to your key as an argument, using -i. For instance,

```python
ssh -L 127.0.0.1:8888:127.0.0.1:8888 -i <SSH_KEYPATH> cc@<FLOATING_IP>
```

Leave this SSH session open. 

Then, run the following cell, which will start a command that does not terminate:
:::

:::{.cell .code}
```
node.run("/home/cc/.local/bin/jupyter notebook --port=8888 --notebook-dir='/home/cc/ml-energy/notebooks/'")
```
:::


:::{.cell}
In the output of the cell above, look for a URL in this format:

http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Copy this URL and open it in a browser. Then, you can run the sequence of notebooks that you'll see there, in order.

If you need to stop and re-start your Jupyter server,

* Use Kernel > Interrupt Kernel twice to stop the cell above
* Then run the following cell to kill whatever may be left running in the background.

Note: If the message 
`The port 8888 is already in use, trying another port.` appears in the output of the above cell, it implies that the local port 8888 is busy i.e. being used by someother process. Note the port the notebook was launched at. `localhost:XXXX`, XXXX is the port of interest. 

Quit the ssh running on the local machine from the cell above and replace it with

```python
ssh -L 127.0.0.1:8888:127.0.0.1:XXXX -i <SSH_KEYPATH> cc@<FLOATING_ID>
```
:::

:::{.cell .code}
```
node.run("sudo killall jupyter-notebook")
```
:::

:::{.cell}
## Release Resources

If you finish with your experimentation before your lease expires,release your resources and tear down your environment by running the following (commented out to prevent accidental deletions).

This section is designed to work as a "standalone" portion - you can come back to this notebook, ignore the top part, and just run this section to delete your reasources
:::

:::{.cell .code}
```
# setup environment - if you made any changes in the top part, make the same changes here
import chi, os
from chi import lease, server, network, share

PROJECT_NAME = "CHI-231095"
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)
username = os.getenv('USER')

lease = lease.get_lease(f"{username}-{NODE_TYPE}")
```
:::


:::{.cell .code}
```
DELETE = False #Default value is False to prevent any accidental deletes. Change it to True for deleting the resources

if DELETE:
    # delete server
    server_id = server.get_server_id(f"{username}-{NODE_TYPE}")
    server.delete_server(server_id)

    # release floating IP
    reserved_fip =  lease.get_reserved_floating_ips(lease["id"])[0]
    ip_info = network.get_floating_ip(reserved_fip)
    chi.neutron().delete_floatingip(ip_info["id"])

    # delete lease
    lease.delete_lease(lease["id"])

    # Delete share
    share.delete_share(my_share)
```
:::
