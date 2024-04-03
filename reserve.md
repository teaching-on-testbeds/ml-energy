:::{.cell}
# Run a single user notebook server on Chameleon

This notebook describes how to run a single user Jupyter notebook server on Chameleon. This allows you to run experiments requiring bare metal access, storage, memory, GPU and compute resources on Chameleon using a Jupyter notebook interface.
:::

:::{.cell}
## Provision the resource

### Check resource availability

This notebook will try to reserve a RTX6000 GPU backed Ubuntu-22.04 on CHI@UC - pending availability. Before you begin, you should check the host calendar at https://chi.uc.chameleoncloud.org/project/leases/calendar/host/ to see what node types are available.

### Chameleon configuration

You can change your Chameleon project name (if not using the one that is automatically configured in the JupyterHub environment) and the site on which to reserve resources (depending on availability) in the following cell.
:::

:::{.cell .code}
```
import chi, os, time
from chi import lease
from chi import server

PROJECT_NAME = "CHI-231095" # change this if you need to
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)
username = os.getenv('USER') # all exp resources will have this prefix
```
:::

:::{.cell .code}
```
image_name="CC-Ubuntu22.04-CUDA"
NODE_TYPE = "gpu_rtx_6000"

chi.set("image", image_name)
```
:::

:::{.cell}
## Reservation
The following cell will create a reservation that begins now, and ends in 8 hours. You can modify the start and end date as needed.
:::

:::{.cell .code}
```
res = []
lease.add_node_reservation(res, node_type=NODE_TYPE, count=1)
lease.add_fip_reservation(res, count=1)

start_date, end_date = lease.lease_duration(days=0, hours=8)
# if you won't start right now - comment the line above, uncomment two lines below
# start_date = '2024-04-02 15:24' # manually define to desired start time 
# end_date = '2024-04-03 01:00' # manually define to desired start time 


l = lease.create_lease(f"{username}-{NODE_TYPE}", res, start_date=start_date, end_date=end_date)
l = lease.wait_for_active(l["id"]) #Comment this line if the lease starts in the future
```
:::

:::{.cell .code}
```
# continue here, whether using a lease created just now or one created earlier
l = lease.get_lease(f"{username}-{NODE_TYPE}")
```
:::

:::{.cell}
## Provisioning resources 

This cell provisions resources. It will take approximately 10 minutes. You can check on its status in the Chameleon web-based UI: https://chi.uc.chameleoncloud.org/project/instances/, then come back here when it is in the READY state.
:::

:::{.cell .code}
```
reservation_id = lease.get_node_reservation(l["id"])
server.create_server(
    f"{username}-{NODE_TYPE}", 
    reservation_id=reservation_id,
    image_name=image_name
)
server_id = server.get_server_id(f"{username}-{NODE_TYPE}")
server.wait_for_active(server_id)
```
:::

:::{.cell}
Associate an IP address with this server:
:::

:::{.cell .code}
```
reserved_fip = server.associate_floating_ip(server_id)
```
:::

:::{.cell}
And wait for it to come up
:::

:::{.cell .code}
```
server.wait_for_tcp(reserved_fip, port=22)
```
:::

:::{.cell}
## Install Stuff

The following cells will install some basic packages for your Chameleon server.
:::

:::{.cell .code}
```
from chi import ssh

node = ssh.Remote(reserved_fip)
```
:::

:::{.cell .code}
```
node.run('sudo apt update')
node.run('sudo apt -y install python3-pip python3-dev')
node.run('sudo pip3 install --upgrade pip')
node.run('sudo apt -y install libcudnn8=8.9.6.50-1+cuda12.2') #Installing appropriate version of cudnn for the installed drivers
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
server.wait_for_tcp(reserved_fip, port=22)
```
:::

:::{.cell .code}
```
node = ssh.Remote(reserved_fip) # note: need a new SSH session to get new PATH
node.run('nvidia-smi')
node.run('nvcc --version')
```
:::

:::{.cell}
## Instal Python packages
:::

:::{.cell .code}
```
node.run('python3 -m pip install --user tensorflow[and-cuda]')
node.run('python3 -m pip install --user numpy')
node.run('python3 -m pip install --user matplotlib')
node.run('python3 -m pip install --user seaborn')
node.run('python3 -m pip install --user librosa')
node.run('python3 -m pip install --user zeus')
```
:::

:::{.cell}
Test your installation - make sure Tensorflow can see the GPU:
:::


:::{.cell .code}
```
node.run('python3 -c \'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))\'')
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
## Run a JupyterHub server

Run the following cell
:::

:::{.cell .code}
```
print('ssh -L 127.0.0.1:8888:127.0.0.1:8888 cc@' + reserved_fip) 
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
ssh -L 127.0.0.1:8888:127.0.0.1:8888 -i <SSH_KEYPATH> cc@<RESERVED_FIP>
```

Leave this SSH session open. 

Then, run the following cell, which will start a command that does not terminate:
:::

:::{.cell .code}
```
node.run("/home/cc/.local/bin/jupyter notebook --port=8888 --notebook-dir='/home/cc/'")
```
:::


:::{.cell}
In the output of the cell above, look for a URL in this format:

http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Copy this URL and open it in a browser. Then, you can run the sequence of notebooks that you'll see there, in order.

If you need to stop and re-start your Jupyter server,

* Use Kernel > Interrupt Kernel twice to stop the cell above
* Then run the following cell to kill whatever may be left running in the background.
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
from chi import lease, server

PROJECT_NAME = PROJECT_NAME
chi.use_site("CHI@UC")
chi.set("project_name", PROJECT_NAME)
username = os.getenv('USER')

lease = chi.lease.get_lease(f"{username}-{NODE_TYPE}")
```
:::


:::{.cell .code}
```
DELETE = False #Default value is False to prevent any accidental deletes. Change it to True for deleting the resources

if DELETE:
    # delete server
    server_id = chi.server.get_server_id(f"{username}-{NODE_TYPE}")
    chi.server.delete_server(server_id)

    # release floating IP
    reserved_fip =  chi.lease.get_reserved_floating_ips(lease["id"])[0]
    ip_info = chi.network.get_floating_ip(reserved_fip)
    chi.neutron().delete_floatingip(ip_info["id"])

    # delete lease
    chi.lease.delete_lease(lease["id"])
```
:::
