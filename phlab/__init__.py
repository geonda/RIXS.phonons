from phlab import model
from phlab import experiment
from phlab import visual
import json
import os

class rixs(object):

    """

    Class rixs exists as a wrapper around\
    both models and experiment objects.

    Args:
        problem_name:  str
                name of the project.
        out_dir: str
                name of the ouptu directory.
        inp_dir: str
                name of the ouptu directory.


    Attributes:
        nmodel: int
            number of models created within this project.
        nexp: int
            number of exp created within this project.
        abs_path: str
            absolute path to the working directory

    """

    def __init__(self,project_name = '', \
                    out_dir = '/_output/', \
                    inp_dir = '/_input/' ):
        super(rixs, self).__init__()
        self.nmodel = 0
        self.nexp = 0
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.abs_path = os.path.abspath('.')




    def decorator_model(function):
        # @wraps(function)
        def wrapper(self,name):

            self.nmodel += 1
            self.name = name
            if name !='':
                base = '{name_}'.format(name_ = self.name)
            else:
                base = 'model_{nmodel_}'.format(nmodel_ = self.nmodel)

            base = os.path.join(self.abs_path,base)

            if not os.path.isdir(base) :
                 os.mkdir(base)

            inp_dir = base+self.inp_dir
            out_dir = base+self.out_dir

            print('creating model : {dir}'.format(dir = base))
            print(inp_dir)

            if not os.path.isdir(inp_dir):
                os.mkdir(inp_dir)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            kwargs={'inp_dir': inp_dir,
                        'out_dir': out_dir,
                        'nmodel' : self.nmodel,
                        'name' : name}

            return function(self,**kwargs)

        return wrapper


    @decorator_model
    def model_gq_phonons_2d(self,**kwargs):
        """

        Model describing interaction of q dependent phonons
        and a single elctornic  level.

        Args:
            name: str
                name of the model

        Note:
            input and output files are located inside  './name/' directory

        Returns:
            model.single_osc(): object
                calls model sub-package

        """
        return model.gq_phonons_2d(**kwargs)



    @decorator_model
    def model_single_osc(self,**kwargs):
        """

        Model describing a harmonic oscillator interacting
        with a single electronic level.

        Args:
            name: str
                name of the model

        Note:
            input and output files are located inside  './name/' directory

        Returns:
            model.single_osc(): object
                calls model sub-package

        """
        return model.single_osc(**kwargs)


    @decorator_model
    def model_double_osc(self, **kwargs):
        """

        Model describing 2D harmonic oscillator which interacts
        with a single electronic level.

        Args:
            name: str
                name of the model

        Note:
            input and output files are located inside  './name/' directory

        Returns:
            model.double_osc(): object
                calls model sub-package

        """

        return model.double_osc(**kwargs)

    @decorator_model
    def model_dist_disp_osc(self, **kwargs):

        """

        Model describing distorted and displaced in the excited-state harmonic oscillator
        which interacts with a single electronic level.

        Args:
            name: str
                name of the model

        Note:
            input and output files are located inside  './name/' directory

        Returns:
            model.dist_disp_osc(): object
                calls model sub-package

        """

        return model.dist_disp_osc(**kwargs)

    def experiment(self,file='', col=[0,1], name = ''):

        """

        Experiment.

        Args:
            file: str
                path to the file with the exp data
            col: list
                [column x ; column y] defines which columns to read from the file
            name: str
                name of the experiment


        Returns:
            experiment.experiment(): object
                calls experiment sub-package

        """

        self.nexp+=1
        return experiment.experiment(expfile = file, \
                                    columns= col, nexp = self.nexp, name = name)


    def visual(self,model_list = [], exp = []):
        """

        Creates visual object within the current project (works space).

        Args:
            model_list:  list
                list of models to plot
            exp: object
                experiment to plot

        Returns:
            visual.plot():  object
                calls visual sub-package

        """
        return visual.plot(model_list = model_list, exp = exp)
