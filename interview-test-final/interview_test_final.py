import logging
import json
from collections import namedtuple
import os

def sort_file_paths(project_name: str):
    # figure out the path of the file we're runnning
    runpath = os.path.realpath(__file__)
    # trim off the bits we know about (i.e. from the root dir of this project)
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    # change directory so we can use relative filepaths
    os.chdir(rundir + project_name)

def load_config():
    run_configuration_file = '../resources/interview-test-final.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration

if __name__ == '__main__':
    # Initialize logging
    logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')

    # Actions: get into working directory, load project config, create dated directories
    sort_file_paths(project_name='interview-test-final')
    run_configuration = load_config()


    # TODO: Load the data by instantiating the FileDataLoader, handle file doesn't exist.
    data_loader = None  # Candidate , instantiate your class here

    # TODO: Do the rest of your work here, or in other classes that are called here.
    model = None  # Candidate, instantiate your class here

    logging.info('Completed program')
