from setuptools import find_packages, setup
from typing import List

edot = "-e ."
def get_requrements(path :str)->List[str]:
    requirements = []
    with open(path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace("\n", "") for requirement in requirements]
        if edot in requirements:
            requirements.remove(edot)
    return requirements

setup( name = "Ujjwal Deep End to End project", 
      version = "0.0.1", 
      author = "Ujjwal Deep", 
      author_email= "ujjwaldeep429@gmail.com", 
      packages = find_packages(),
      install_requires = get_requrements('requirements.txt') )