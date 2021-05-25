---
title: Manage ML Process
weight: 10
---
<!-- Just need a katex comment to generate katex -->

{{< katex >}} {{< /katex >}}

# Managing ML Process

## Using Shell Scripts

Shell scripts can help automate many low-level processes in the process of developing machine learning models. Here are some examples that you can use shell scripts for:
- Periodically pull data sets from Hadoop to local desktop and transfer the data set to your AWS
- Clean up unnecessary artifacts, such as data files, in a project folder before checking the code into the kit repository.

Here is an example of a shell script that I use to quickly make update to my git repository.  With this, instead of `git add`, `git commit -m 'commit message'`, followed by `git push`, I just do `lap 'commit message` to finish my push to the current branch. 

```bash
# >>> lazy git push >>>
function lap() {
    git pull
    git add .
    git commit -a -m "$1"
    git push
}
```

### Some useful tools that shine in Shell

First, we can use a more powerful shell with [oh my zsh][1]

Here are some other tools that I personally use at least once a day: 
[rsync][2]: compressing and then uploading large files across different instances
[tmux][3]: keep my processes running even after I close the server connection 
[htop][4]: a more comprehensive display of the top processes 
[gpustat][5]: a process statistics that focuses on the GPU usage 
[prettier][6] (work with node.js): for code [linting][7] in almost any language. Setup guide [here][8]. 

## Using git

There are many workflows around the code check-in process that makes ML development significantly easier to manage. 

One example is to clean up the code formatting before committing the code 
[Lint code before commit][9]

Here are a few other common tools and usage

### Large File

If he accidentally committed a large file and want to remove it completely from the history of the git repository, [BFG][10] is what you need. Here is the official [site][11].

[Use BFG to remove large file and clean cumbersome git history][12]

### Use `.gitignore`

[Useful set of gitignore][13]
[Ignore an existing file][14]

## Why is a mature ML development process important

A good paper that summarizes challenges in the development and deployment of machine learning systems.
[Hidden Technical Debt in Machine Learning Systems][15]

Here is an example of a mature model deployment process summarized in a Udemy course. The specific implementation does not work for all the projects, but the framework is robust and easy to follow 
[ A mature model deployment process ][16]

## Project Structures

The first step to make the development of ML models easier to manage is to follow a standardized way to set up each project.

[Cookie Cutter project structure][17]

## Docker for ML Development and Deployment

An even more powerful tool to standardize the process is Docker.

**Motivation**

I found this [docker container][18] to give a good starter project template. Creating this a docker container for your project does a several things:
1. Create a clean environment for model development that is separate from all the other system set ups
2. Automatically set up a GPU to work with tensorflow. In this docker image, `Dockerfile.gpu` contains the settings for GPU machines. Dockerfile.cpu contains settings to be that can be used in production for non-GPU machines.

**Here is how to set it up. **

1. Install Docker for Mac [Docker for Mac][19]
2. Optional – if you plan to use GPU, install `nvidia-docker`. Note that it cannot be installed on Mac. Here is a helpful [guide][20].
```bash
# a clean installation of docker ce
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo rm -rf /var/lib/docker
sudo apt-get autoclean
sudo apt-get update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
sudo systemctl status docker

# use docker without sudo
sudo usermod -aG docker ${USER}
su - ${USER}
id -nG

# install nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo docker run --runtime=nvidia --rm nvidia/cuda:10.1-base nvidia-smi
```
4. In a `system`-level virtual/conda environment `pip install cookiecutter`
	1. Cookie cutter is a package that makes it easy to create templates for different projects. Customization variables such as `{{ project_name }}` can be defined in the templates to allow customized variable names for each project.
5. Enter your git repository folder, where all of your other projects are located.
6. Use cookie cutter module installed just now to setup the project `cookiecutter git@github.com:docker-science/cookiecutter-docker-science.git`. It will be asked as series of questions to customize your project.
	1. You will be asked a series of questions to help setup the environment. Your answers to this question will be used to configure a range of settings from the name of the project folder to the directories.
	2. The options provided in the brackets are the default. If you choose to go as a default, simply press ENTER in the command line.
	3. The [base][21] docker [image][22] used for this project is a good default.
7. After answering all the questions, the new project folder will be created. 
8. Start the docker app on your current host. 
9. enter the newly created project folder and type `make init` to setup up the container, followed by `make create-container` to start the development environment. 
	1. The first time you set up your environment, you will see various packages being downloaded. This is because Docker needs to install all the necessary packages to set up a separate operating system within your computer.
	2. The default port for notebook is `8888`. You can override with `make create-container JUPYTER_HOST_PORT=9900`
10. After finishing `make create-container` for the first time, you can enter the container again with `make start-container` in the project directory.
11. `docker ps` to view the host port that is being forwarded in order to access Jupyter notebook.

[1]:	https://ohmyz.sh
[2]:	https://www.digitalocean.com/community/tutorials/how-to-use-rsync-to-sync-local-and-remote-directories
[3]:	https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/
[4]:	https://delightlylinux.wordpress.com/2014/03/24/htop-a-better-process-viewer-then-top/
[5]:	https://github.com/wookayin/gpustat
[6]:	https://prettier.io
[7]:	https://en.wikipedia.org/wiki/Lint_(software)
[8]:	https://prettier.io/docs/en/install.html
[9]:	https://prettier.io/docs/en/precommit.html
[10]:	https://formulae.brew.sh/formula/bfg
[11]:	https://rtyley.github.io/bfg-repo-cleaner/
[12]:	https://www.phase2technology.com/blog/removing-large-files-git-bfg
[13]:	https://github.com/github/gitignore
[14]:	https://stackoverflow.com/questions/1274057/how-to-make-git-forget-about-a-file-that-was-tracked-but-is-now-in-gitignore
[15]:	https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
[16]:	https://christophergs.github.io/machine%20learning/2019/03/17/how-to-deploy-machine-learning-models/
[17]:	https://drivendata.github.io/cookiecutter-data-science/#directory-structure
[18]:	https://github.com/docker-science/cookiecutter-docker-science
[19]:	https://store.docker.com/editions/community/docker-ce-desktop-mac
[20]:	https://medium.com/@linhlinhle997/how-to-install-docker-and-nvidia-docker-2-0-on-ubuntu-18-04-da3eac6ec494
[21]:	https://docs.docker.com/develop/develop-images/baseimages/
[22]:	https://hub.docker.com/r/manifoldai/orbyter-ml-dev