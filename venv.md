# Virtual env instructions

First do `cd path_env` where path_env is the path of the folder where you want to create your virtual env. Then run `python3 -m venv projectname` or `python -m venv projectname` where projectname is the name of the virtual env.

Run `source path_env/projectname/bin/activate` to activate your virtual env. Run `deactivate` when you have finished using your virtual env.

Then go to your project folder with `cd project_folder`.

Run `pip install -r requirements.txt` and then `ipython kernel install --user --name=projectname` to add your virtual env to the jupyter kernels.