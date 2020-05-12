import json
import config as cg
import numpy as np
class inputp(object):
    """docstring for read_input."""
    def __init__(self, nproblems,ninputs):
        super(inputp, self).__init__()
        self.nproblems=nproblems
        self.ninputs=ninputs
        self.dict_problem_file=cg.dict_problem_file
        self.dict_problem=self.upload_dict()
        self.dict_input=cg.dict_input_file+'_'+str(self.dict_problem['type_calc'])+'_'\
                +str(self.dict_problem['type_problem'])+'_'+str(self.nproblems)+\
                                                '_'+str(self.ninputs)+'.json'
    def upload_dict(self):
        try :
            with open(self.dict_problem_file) as fp:
                dict=json.load(fp)
            return dict
        except:
            print('error: no dict')
    def upload_dict_soft(self):
        try :
            with open(self.dict_input) as fp:
                dict=json.load(fp)
            return dict
        except:
            print('no input file found')
            pass
    def update(self,name,discrip,dict):
            print(discrip+str(dict[name]))
            try:
                value=float(input())
                dict[name]=value
            except ValueError:
                pass
            return dict
    def create(self,name,discrip,dict):
        print(discrip)
        try:
            value=float(input())
            dict[name]=value
            return dict
        except ValueError:
            print('value error')
    def check_problem_type(self):
        list_names_fc=['max number of final state vibrational levels: ', \
         'max number of intermediate state vibrational levels: ',\
         'excitation energy (eV): ','incoming photon energy (eV): ',\
        'inverse core-hole lifetime (eV): ',\
        'inverse phonon lifetime (eV): ',\
        'exp broadening (eV):' ]
        list_features_fc=['nf','nm','energy_ex','omega_in',\
        'gamma','gamma_ph','alpha_exp']

        list_names_gf=['max number of final state vibrational levels: ',\
        'max number of time steps: ', 't_max: ',\
         'excitation energy (eV): ','incoming photon energy (eV): ',\
        'inverse core-hole lifetime (eV): ',\
        'inverse phonon lifetime (eV): ',\
        'exp broadening (eV):' ]
        list_features_gf=['nf','nstep','maxt','energy_ex','omega_in',\
        'gamma','gamma_ph','alpha_exp']


        #self.dict_problem=self.upload_dict()
        list_names_main,list_features_main=[],[]
        for i in range(int(self.dict_problem['vib_space'])):
            list_names_main.append(str(i)+' mode electron phonon coupling (eV): ')
            list_names_main.append(str(i)+' mode phonon energy (eV): ')
            list_features_main.append('coupling'+str(i))
            list_features_main.append('omega_ph'+str(i))
        if self.dict_problem['type_calc']=='dd':
            list_names_main.extend(['excited state phonon energy (eV): '])
            list_features_main.extend(['omega_ph_ex'])
        if self.dict_problem['type_problem']=='rixs_q':
            list_names_main.extend(['number of q points: '])
            list_features_main.extend(['nq'])
            list_names_main.extend(['rixs q: '])
            list_features_main.extend(['qx'])
        if self.dict_problem['method']=='fc':
            list_names_main.extend(list_names_fc)
            list_features_main.extend(list_features_fc)
        elif self.dict_problem['method']=='gf':
            list_names_main.extend(list_names_gf)
            list_features_main.extend(list_features_gf)
        # print(list_names_main)

        return list_features_main,list_names_main
    def create_input(self):
        list_rixs,list_rixs_names=self.check_problem_type()
        dict_temp=self.upload_dict_soft()
        if self.dict_problem['type_problem']=='rixs_q':
            print('to modify coupling constant check phonon_info.py')
        if dict_temp:
            for name,discrip in zip(list_rixs,list_rixs_names):
                dict_temp=self.update(name,discrip,dict_temp)
        else:
            dict_temp={}
            for name,discrip in zip(list_rixs,list_rixs_names):
                dict_temp=self.create(name,discrip,dict_temp)
        # print(dict_temp)
        for i in range(int(self.dict_problem['vib_space'])):
            dict_temp['g'+str(i)]=(dict_temp['coupling'+str(i)]\
                                            /dict_temp['omega_ph'+str(i)])**2
        # print(dict_temp)
        with open(self.dict_input, 'w') as fp:
            json.dump(dict_temp,fp,indent=1)
        print(' >>>>> input created ')
        print(dict_temp)
        return dict_temp
    def scan_input(self):
        dict_scan={}
        # with open('input.json', 'w') as fp:
        #     json.load(dict_temp,fp)
        print('enter number of runs')
        dict_scan['nruns']=int(input())
        print('input min coupling')
        gmin=float(input())
        print('input max coupling')
        gmax=float(input())
        dict_scan['coupling']=list(np.linspace(gmin,gmax,dict_scan['nruns']))
        with open(cg.dict_scan_file, 'w') as fp:
            json.dump(dict_scan,fp,indent=1)
        return dict_scan

    def update_total(self,dict_total):
        for i in range(int(dict_total['problem']['vib_space'])):
            dict_total['input']['g'+str(i)]=(dict_total['input']['coupling'+str(i)]\
                                            /dict_total['input']['omega_ph'+str(i)])**2
        return dict_total
