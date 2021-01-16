""" Learning notes
multiprocessing is a package for the Python language which supports the spawning 
of processes using the API of the standard library’s threading module.
Objects can be transferred between processes using pipes or multi-producer/ 
multi-consumer queues.Objects can be shared between processes using a server 
process or (for simple data) shared memory. Equivalents of all the synchronization 
primitives in threading are available.
A Pool class makes it easy to submit tasks to a pool of worker processes.
Examples: https://pymotw.com/2/multiprocessing/basics.html
Difference between pool and process: https://www.ellicium.com/python-multiprocessing-pool-process/
The pool distributes the tasks to the available processors using a FIFO scheduling. 
It works like a map-reduce architecture. It maps the input to the different 
processors and collects the output from all the processors. After the execution 
of code, it returns the output in form of a list or array.  It waits for all 
the tasks to finish and then returns the output. The processes in execution are 
stored in memory and other non-executing processes are stored out of memory. Use 
pool when there are many tasks and few resources; Use process when number of 
tasks is few and you can assign one task to one process

Click is a Python package for creating beautiful command line interfaces in a 
composable way with as little code as necessary. It’s the “Command Line 
Interface Creation Kit”. It’s highly configurable but comes with sensible 
efaults out of the box. It aims to make the process of writing command line 
tools quick and fun while also preventing any frustration caused by the 
inability to implement an intended CLI API.
Click in three points:
    Arbitrary nesting of commands
    Automatic help page generation
    Supports lazy loading of subcommands at runtime
A Simple Example is here:
###### Program ######
import click
@click.command()
@click.option("--count", default=1, help="Number of greetings.")
@click.option("--name", prompt="Your name", help="The person to greet.")
def hello(count, name):
    #Simple program that greets NAME for a total of COUNT times.
    for _ in range(count):
        click.echo(f"Hello, {name}!")
if __name__ == '__main__':
    hello()
###### Output ######
$ python hello.py --count=3
Your name: Bose
Hello, Bose!
Hello, Bose!
Hello, Bose!

Logging in Python : https://www.loggly.com/ultimate-guide/python-logging-basics/
The module provides a way for applications to configure different log handlers 
and a way of routing log messages to these handlers. This allows for a highly 
flexible configuration that can deal with a lot of different use cases.
To emit a log message, a caller first requests a named logger. The name can be 
used by the application to configure different rules for different loggers. 
This logger then can be used to emit simply-formatted messages at different 
log levels (DEBUG, INFO, ERROR, etc.), which again can be used by the application 
to handle messages of higher priority different than those of a lower priority. 
While it might sound complicated, it can be as simple as this:
============
import logging
log = logging.getLogger("my-logger")
log.info("Hello, world")
============
Logging from Modules
A well-organized Python application is likely composed of many modules. Reusable
modules available from pypi  emit log messages as a best practice and should not 
configure how those messages are handled. That is the responsibility of the 
application. The only responsibility modules have is to make it easy for the 
application to route their log messages. For this reason, it is a convention 
for each module to simply use a logger named like the module itself. This makes 
it easy for the application to route different modules differently, while also 
keeping log code in the module simple. The module just needs two lines to set 
up logging, and then use the named logger:
===============
import logging
log = logging.getLogger(__name__)
def do_something():
    log.debug("Doing something!")
===============
Configuring Logging
Your main application should configure the logging subsystem so log messages go 
where they should. The Python logging module provides a large number of ways to 
fine-tune this, but for almost all applications, the configuration can be very 
simple. In general, a configuration consists of adding a Formatter and a Handler 
to the root logger. Because this is so common, the logging module provides a 
utility function called basicConfig that handles a majority of use cases
Example 1: send log messages to stdout or stderr and have systemd forward 
the messages to journald and syslog.
==========
import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
exit(main())
=============
That's it. The application will now log all messages with level INFO or above 
to stderr, using a simple format:
ERROR:the.module.name:The log message
The application can even be configured to include DEBUG messages, or maybe 
only ERROR, by setting the LOGLEVEL environment variable.
"""

import os
from multiprocessing import Pool
import glob

import click
import logging
import pandas as pd


#Modification to avoid ModuleNotFoundError: No module named 'src'
#https://stackoverflow.com/questions/16480898/receiving-import-error-no-module-named-but-has-init-py
#Just a note for anyone who arrives at this issue, using what Gus E showed in 
# the accept answer and some further experience I've found the following to be 
# very useful to ensure that I can run my programs from the command-line on my 
# machine or on another colleague's should the need arise.
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import sys
#sys.path.append("/home/user/DMML/CodeAndRepositories/MMGTVSeg")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src
from src.resampling.resampling import Resampler


# Default paths

path_in = 'data/hecktor_test/hecktor_nii_test/'
path_out = 'data/hecktor_test/resampled_test/'
path_bb = 'data/hecktor_test/bbox_test.csv'

# path_in = 'data/hecktor_train/hecktor_nii/'
# path_out = 'data/hecktor_train/resampled/'
# path_bb = 'data/hecktor_train/bbox.csv'

# path_in = '/home/user/DMML/Data/HeadNeck_PET_CT/HecktorData/hecktor_train/hecktor_nii/'
# path_out = '/home/user/DMML/Data/HeadNeck_PET_CT/HecktorData/hecktor_train/resampled/'
# path_bb = '/home/user/DMML/Data/HeadNeck_PET_CT/HecktorData/hecktor_train/bbox.csv'


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
@click.option('--cores',
              type=click.INT,
              default=12,
              help='The number of workers for parallelization.')
@click.option('--resampling',
              type=click.FLOAT,
              nargs=3,
              default=(1, 1, 1),
              help='Expect 3 positive floats describing the output '
              'resolution of the resampling. To avoid resampling '
              'on one or more dimension a value of -1 can be fed '
              'e.g. --resampling 1.0 1.0 -1 will resample the x '
              'and y axis at 1 mm/px and left the z axis untouched.')
@click.option('--order',
              type=click.INT,
              nargs=1,
              default=3,
              help='The order of the spline interpolation used to resample')
def main(input_folder, output_folder, bounding_boxes_file, cores, resampling,
         order):
    """ This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation
        of degree --order (default=3) and the segmentation are resampled
        by nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    logger = logging.getLogger(__name__)
    logger.info('Resampling')

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print('resampling is {}'.format(str(resampling)))
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    files_list = [
        f for f in glob.glob(input_folder + '/**/*.nii.gz', recursive=True)
    ]
    resampler = Resampler(bb_df, output_folder, order, resampling=resampling)
    with Pool(cores) as p:
        p.map(resampler, files_list)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
