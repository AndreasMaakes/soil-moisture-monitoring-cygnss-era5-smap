# README

## Setting up the virtual enviroment

Head to the root folder of the project (replace the path below with your local path):

```bash
cd ~/Desktop/my_project
```

Create the virtual enviroment:

```bash
python -m venv venv
```

Activate the environment:

For macOS:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

## Installing the dependencies

When the user is inside the venv and wants to install the dependencies for the project, they have to use these commands:

```bash
pip install -r requirements.txt
```
