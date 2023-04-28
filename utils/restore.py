# imported from https://github.com/Chakazul/Lenia/blob/master/Python/LeniaND.py

import numpy as np                    
from fractions import Fraction

DIM = 2
DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}


class Board:
    def __init__(self, size=[0]*DIM):
        self.names = ['', '', '']
        self.params = {'R':13, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1}
        self.param_P = 0
        self.cells = np.zeros(size)

    @classmethod
    def from_values(cls, cells, params=None, names=None):
        self = cls()
        self.names = names.copy() if names is not None else None
        self.params = params.copy() if params is not None else None
        self.cells = cells.copy() if cells is not None else None
        return self

    @classmethod
    def from_data(cls, data):
        self = cls()
        self.names = [data.get('code',''), data.get('name',''), data.get('cname','')]
        self.params = data.get('params')
        if self.params:
            self.params = self.params.copy()
            self.params['b'] = Board.st2fracs(self.params['b'])
        self.cells = data.get('cells')
        if self.cells:
            if type(self.cells) in [tuple, list]:
                self.cells = ''.join(self.cells)
            self.cells = Board.rle2arr(self.cells)
        return self.names[1], self.params, self.cells


    def params2st(self):
        params2 = self.params.copy()
        params2['b'] = '[' + Board.fracs2st(params2['b']) + ']'
        return ','.join(['{}={}'.format(k,str(v)) for (k,v) in params2.items()])

    def long_name(self):
        # return ' | '.join(filter(None, self.names))
        return '{0} - {1} {2}'.format(*self.names)

    @staticmethod
    def ch2val(c):
        if c in '.b': return 0
        elif c == 'o': return 255
        elif len(c) == 1: return ord(c)-ord('A')+1
        else: return (ord(c[0])-ord('p')) * 24 + (ord(c[1])-ord('A')+25)

    @staticmethod
    def val2ch(v):
        if v == 0: return ' .'
        elif v < 25: return ' ' + chr(ord('A')+v-1)
        else: return chr(ord('p') + (v-25)//24) + chr(ord('A') + (v-25)%24)

    @staticmethod
    def _recur_drill_list(dim, lists, row_func):
        if dim < DIM-1:
            return [Board._recur_drill_list(dim+1, e, row_func) for e in lists]
        else:
            return row_func(lists)

    @staticmethod
    def _recur_join_st(dim, lists, row_func):
        if dim < DIM-1:
            return DIM_DELIM[DIM-1-dim].join(Board._recur_join_st(dim+1, e, row_func) for e in lists)
        else:
            return DIM_DELIM[DIM-1-dim].join(row_func(lists))

    @staticmethod
    def _append_stack(list1, list2, count, is_repeat=False):
        list1.append(list2)
        if count != '':
            repeated = list2 if is_repeat else []
            list1.extend([repeated] * (int(count)-1))

    @staticmethod
    def _recur_get_max_lens(dim, list1, max_lens):
        max_lens[dim] = max(max_lens[dim], len(list1))
        if dim < DIM-1:
            for list2 in list1:
                Board._recur_get_max_lens(dim+1, list2, max_lens)

    @staticmethod
    def _recur_cubify(dim, list1, max_lens):
        more = max_lens[dim] - len(list1)
        if dim < DIM-1:
            list1.extend([[]] * more)
            for list2 in list1:
                Board._recur_cubify(dim+1, list2, max_lens)
        else:
            list1.extend([0] * more)

    @staticmethod
    def rle2arr(st):
        stacks = [[] for dim in range(DIM)]
        last, count = '', ''
        delims = list(DIM_DELIM.values())
        st = st.rstrip('!') + DIM_DELIM[DIM-1]
        for ch in st:
            if ch.isdigit(): count += ch
            elif ch in 'pqrstuvwxy@': last = ch
            else:
                if last+ch not in delims:
                    Board._append_stack(stacks[0], Board.ch2val(last+ch)/255, count, is_repeat=True)
                else:
                    dim = delims.index(last+ch)
                    for d in range(dim):
                        Board._append_stack(stacks[d+1], stacks[d], count, is_repeat=False)
                        stacks[d] = []
                    #print('{0}[{1}] {2}'.format(last+ch, count, [np.asarray(s).shape for s in stacks]))
                last, count = '', ''
        A = stacks[DIM-1]
        max_lens = [0 for dim in range(DIM)]
        Board._recur_get_max_lens(0, A, max_lens)
        Board._recur_cubify(0, A, max_lens)
        return np.asarray(A)

    @staticmethod
    def fracs2st(B):
        return ','.join([str(f) for f in B])

    @staticmethod
    def st2fracs(st):
        return [Fraction(st) for st in st.split(',')]



