import os
import urllib.request
import io
import gzip
import shutil
import tarfile

# create directory

os.makedirs('notebooks', exist_ok=True)

# list all notebooks

repo = 'simonwm/tacco_examples'
branch = 'main'
repo_url = f'https://github.com/{repo}/tree/{branch}/notebooks'
raw_url = f'https://raw.githubusercontent.com/{repo}/{branch}/notebooks'
with urllib.request.urlopen(repo_url) as f:
    response = f.read().decode('utf-8')

pieces = response.split('"notebooks/')[1:]
basenames = [p.split('.ipynb"')[0] for p in pieces]

# download all notebooks

for basename in basenames:
    with urllib.request.urlopen(f'{raw_url}/{basename}.ipynb') as f:
        with open(f'notebooks/{basename}.ipynb', 'wb') as g:
            shutil.copyfileobj(f, g)

# create new examples.rst file

with open('examples.rst', 'w') as f:
    notebooks = "\n   ".join([f'notebooks/{b}' for b in basenames])
    f.write(f'''Examples
--------

The notebooks for the examples and the workflow to prepare the necessary datasets are available in the `example repository <https://github.com/simonwm/tacco_examples>`_.

.. nbgallery::
   {notebooks}
''')
