import os
import pickle
import re
import pandas as pd
import numpy as np
from hook_interface import HookInterface
from git import Repo
from unidiff import PatchSet

MODEL_PATH = os.path.dirname(__file__)
print(MODEL_PATH)


class JITLineHook(HookInterface):
    def __init__(self):
        super(JITLineHook, self).__init__()
        self.repo = Repo('./')
        self.metrics = None
        self.clf = None

    def run_model(self):
        changed_code = self.collect_codes()
        codes = self.prepare_data(changed_code, remove_python_common_tokens=True)
        self.load_change_metrics_df('apache_metrics_kamei.csv')
        c = ''
        # try:
        #     metrics = self.metrics[self.metrics['commit_id'] == c].drop(columns=['commit_id']).to_numpy(dtype=np.float32)
        # except IndexError:
            # commit id not in commit metric set
        dim = self.metrics[self.metrics['commit_id'] == c].drop(columns=['commit_id']).shape[1]
        metrics = np.zeros((1, dim), dtype=np.float)
        with open(os.path.join(MODEL_PATH, 'count_vect.pkl'), 'rb') as fp:
            count_vect = pickle.load(fp)
        train_feature = self.get_combined_df(codes, metrics, count_vect)
        with open(os.path.join(MODEL_PATH, 'clf.pkl'), 'rb') as fp:
            self.clf = pickle.load(fp)
        pred = self.train_eval_model(train_feature)
        print(pred)
        exit(-1)

    def load_change_metrics_df(self, metrics_file):
        self.metrics = pd.read_csv(os.path.join(MODEL_PATH, metrics_file))
        self.metrics = self.metrics.drop(
            ['author_date', 'bugcount', 'fixcount', 'revd', 'tcmt', 'oexp', 'orexp', 'osexp', 'osawr', 'project',
             'buggy', 'fix'],
            axis=1, errors='ignore')
        self.metrics = self.metrics[['commit_id', 'la', 'ld', 'nf', 'nd', 'ns', 'ent',
                                     'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp']]
        self.metrics = self.metrics.fillna(value=0)

    def collect_codes(self):
        # list of list of dictionaries with keys 'added_code' and 'removed_code'. values are strings of each line
        codes = []
        for m in self.modifieds:
            added_code = []
            removed_code = []
            diff = self.repo.git.diff('--cached', m)
            patch_set = PatchSet(diff)
            for hunk in patch_set[0]:
                for line in hunk:
                    if line.is_added:
                        added_code.append(str(line).strip()[1:])
                    if line.is_removed:
                        removed_code.append(str(line).strip()[1:])
            m_dict = {'added_code': added_code, 'removed_code': removed_code}
            codes.append(m_dict)
        return codes

    @staticmethod
    def preprocess_code_line(code, remove_python_common_tokens=False):
        python_common_tokens = ['abs', 'delattr', 'hash', 'memoryview', 'set', 'all', 'dict', 'help', 'min', 'setattr',
                                'any', 'dir', 'hex', 'next', 'slice', 'ascii', 'divmod', 'id', 'object', 'sorted',
                                'bin', 'enumerate', 'input', 'oct', 'staticmethod', 'bool', 'eval', 'int', 'open',
                                'str', 'breakpoint', 'exec', 'isinstance', 'ord', 'sum', 'bytearray', 'filter',
                                'issubclass', 'pow', 'super', 'bytes', 'float', 'iter', 'print', 'tuple', 'callable',
                                'format', 'len', 'property', 'type', 'chr', 'frozenset', 'list', 'range', 'vars',
                                'classmethod', 'getattr', 'locals', 'repr', 'zip', 'compile', 'globals', 'map',
                                'reversed', '__import__', 'complex', 'hasattr', 'max', 'round', 'False', 'await',
                                'else', 'import', 'passNone', 'break', 'except', 'in', 'raise', 'True', 'class',
                                'finally', 'is', 'return', 'and', 'continue', 'for', 'lambda', 'try', 'as', 'def',
                                'from', 'nonlocal', 'while', 'assert', 'del', 'global', 'not', 'with', 'async', 'elif',
                                'if', 'or', 'yield', 'self']

        code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(
            ']', ' ').replace('.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')
        code = re.sub('``.*``', '<STR>', code)
        code = re.sub("'.*'", '<STR>', code)
        code = re.sub('".*"', '<STR>', code)
        code = re.sub('\d+', '<NUM>', code)

        if remove_python_common_tokens:
            new_code = ''

            for tok in code.split():
                if tok not in python_common_tokens:
                    new_code = new_code + tok + ' '

            return new_code.strip()

        else:
            return code.strip()

    def load_data(self, code_change, use_text=True, remove_python_common_tokens=False):
        if use_text:
            added_code_list = []
            removed_code_list = []

            for i in range(0, len(code_change)):
                ch = code_change[i]
                added_code = ch['added_code']
                removed_code = ch['removed_code']
                if len(added_code) > 0:
                    for code in added_code:
                        if code.startswith("#"):
                            continue
                        added_code_list.append(self.preprocess_code_line(code, remove_python_common_tokens))
                if len(removed_code) > 0:
                    for code in removed_code:
                        if code.startswith("#"):
                            continue
                        removed_code_list.append(self.preprocess_code_line(code, remove_python_common_tokens))
            added_code = ' \n '.join(list(set(added_code_list)))
            removed_code = ' \n '.join(list(set(removed_code_list)))
            return added_code, removed_code

    def prepare_data(self, code_change, use_text=True, remove_python_common_tokens=False):
        if use_text:
            all_added_code, all_removed_code = self.load_data(code_change, use_text=use_text,
                                                              remove_python_common_tokens=remove_python_common_tokens)
            combined_code = [all_added_code + ' ' + all_removed_code]
            return combined_code

    @staticmethod
    def get_combined_df(code, metrics, count_vect):
        code_change_arr = count_vect.transform(code).astype(np.int16).toarray()
        final_features = np.concatenate((code_change_arr, metrics), axis=1)
        return final_features

    def train_eval_model(self, train_feature):
        prob = self.clf.predict_proba(train_feature)
        return prob[0, 1]


if __name__ == '__main__':
    hook = JITLineHook()
    hook.run_model()
