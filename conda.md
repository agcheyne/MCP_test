# Installing miniconda
From https://docs.anaconda.com/miniconda/install/

## install
These four commands download the latest 64-bit version of the Linux installer, rename it to a shorter file name, silently install, and then delete the installer:

**Note: we should not install in home, change the "```<WORKDISK>```" path here to your workdisk area instead**

for example mine would be <WORKDISK> -> /w/work5/home/andrewc

```bash
mkdir -p <WORKDISK>/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O <WORKDISK>/miniconda3/miniconda.sh
bash <WORKDISK>/miniconda3/miniconda.sh -b -u -p <WORKDISK>/miniconda3
rm <WORKDISK>/miniconda3/miniconda.sh
```
Run the following to activate conda
```bash
source <WORKDISK>/miniconda3/bin/activate
```
This last line didn't work for me, I had to:
>cd <WORKDISK>/andrewc/miniconda3/bin/
>
>./activate


Initialise and update conda.
  ```bash
conda init --all
```
```bash
conda update -n base -c defaults conda
```


## First environment
Conda uses a central package cache (e.g., ~/.conda/pkgs) to store package files. When you install numpy in multiple environments, Conda reuses these cached files rather than downloading or unpacking them repeatedly.
As such, you want to create isolated environments for your projects. **Avoid installing packages globally in the base environment**.

```bash
conda create -n myenv python=3.9
```
```bash
conda activate myenv
```
Deactivate when you're done:
```bash
conda deactivate
```

## Environment management
```bash
conda env list
```

```bash
conda remove -n myenv --all
```

```bash
conda env export > environment.yml
```

```bash
conda env create -f environment.yml
```

```bash
conda env list
```

### Example for a generic package or common things for me
```bash
conda create -n myenv python=3.9
```
```bash
conda activate myenv
```
```bash
conda install numpy pandas matplotlib jupyterlab ipykernel
```

## Document Your Workflows
As you set up environments and install packages, keep track of:

Which environments you create (e.g., myenv for project X).
The packages installed and their versions.
Any external tools or configurations (e.g., adding to PATH).
