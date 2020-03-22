from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
import numpy as np
import sympy as sp
from sympy.abc import x,y,c
from sympy.utilities.lambdify import lambdastr
import argparse
from tqdm import tqdm
from sympy import tan, cos, sin, asin, acos, atan, asinh, acosh, atanh, sinh, cosh, tanh, log
import signal
import time
import os
from multiprocessing import Process, Manager


def isOper(ch):
    if ch in ['+', '-', '*', '/', '**', '(', ')','tan', 'cos', 'sin', 'exp', 'log', 'sqrt', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']:
        return True
    return False
    
def getOperOrder(ch):
    prec = {')': 4, 'tan': 4, 'cos': 4, 'sin': 4, '**': 4, '*': 3, '/': 3, '+': 2, '-': 2, '(': 1, 'exp':4, 'log':4, 'sqrt':4, 'asin':4, 'acos':4, 'atan':4, 'sinh':4, 'cosh':4, 'tanh':4, 'asinh':4, 'acosh':4, 'atanh':4}
    return prec[ch]

def infix_to_prefix(infix_expr):

    prec = {')': 4, 'tan': 4, 'cos': 4, 'sin': 4, '**': 4, '*': 3, '/': 3, '+': 2, '-': 2, '(': 1, 'exp':4, 'log':4, 'sqrt':4, 'asin':4, 'acos':4, 'atan':4, 'sinh':4, 'cosh':4, 'tanh':4, 'asinh':4, 'acosh':4, 'atanh':4,'$':5}

    prefix_expr = []
    s = Stack()
    infix_list = []
    if infix_expr in [['('],['0']] or infix_expr[-1] == '#':
#        print('can\'t integrate')
        return 0
    for item in reversed(infix_expr):
        if item not in prec.keys():
            prefix_expr.append(item)
        elif item == ')':
            s.push(item)
        elif item == '(':
            while s.peek() != ')':
                prefix_expr.append(s.pop())
            s.pop()
        else:
            while (not s.isEmpty())\
                    and s.peek() != ')'\
                    and prec[s.peek()] > prec[item]:
                prefix_expr.append(s.pop())
            s.push(item)
    while not s.isEmpty():
        prefix_expr.append(s.pop())
    prefix_expr.reverse()
    return ' '.join(prefix_expr)
#
#def inorder(tree):
#    str = []
#    left = []
#    right = []
#    if tree != None:
#        left = inorder(tree.getLeftChild())
#        str.append(tree.getRootVal())
#        right = inorder(tree.getRightChild())
#    return left+str+right


def InorderTree(tree, res):
    if not tree:
        return
    if tree.leftChild:
        if isOper(tree.leftChild.key) and getOperOrder(tree.leftChild.key) < getOperOrder(tree.key):
            res.append('(')
            InorderTree(tree.leftChild, res)
            res.append(')')
        else:
            InorderTree(tree.leftChild, res)
    res.append(tree.key)
    if tree.rightChild:
        if isOper(tree.rightChild.key) and getOperOrder(tree.rightChild.key) <= getOperOrder(tree.key):
            res.append('(')
            InorderTree(tree.rightChild, res)
            res.append(')')
        elif tree.key in ['tan', 'cos', 'sin', 'exp', 'log', 'sqrt', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']:
            res.append('(')
            InorderTree(tree.rightChild, res)
            res.append(')')
        else:
            InorderTree(tree.rightChild, res)



def string_to_list_c(str):
    list = []
    i = 10
    while i < len(str):
        if str[i] in ['(',')']:
            list.append(str[i])
            i += 1
        elif str[i] == ' ':
            i += 1
        elif str[i] in ['+','/','x']:
            list.append(str[i])
            i += 1
        elif str[i] in ['1','2','3','4','5','6','7','8','9','0']:
            j = 1
            while str[i+j] in ['1','2','3','4','5','6','7','8','9','0']:
                j += 1
            list.append(str[i:i+j])
            i += j
        elif str[i] == '-':
            if str[i-1] not in ['1','2','3','4','5','6','7','8','9','0']:
                if str[i+1] in ['1','2','3','4','5','6','7','8','9','0']:
                    j = 1
                    while str[i+1+j] in ['1','2','3','4','5','6','7','8','9','0']:
                        j += 1
                    list.append('(')
                    list.append(str[i:i+j+1])
                    list.append(')')
                    i += j+1
                elif str[i+1] == 'x':
                    list.append('(')
                    list.append('-1')
                    list.append(')')
                    list.append('*')
                    i += 1
                else:
                    list.append(str[i])
                    i += 1
            else:
                list.append(str[i])
                i += 1
        elif str[i] == '*':
            if str[i+1] == '*':
                list.append('**')
                i += 2
            else:
                list.append('*')
                i += 1
        elif str[i] == 's':
            if str[i+3] == 'h':
                list.append('sinh')
                i += 4
            elif str[i+1] == 'q':
                list.append('sqrt')
                i += 4
            elif str[i+2] == 'n':
                list.append('sin')
                i += 3
            else:
                list.append('#')
                break
        elif str[i] == 'c':
            if i+3 < len(str) and str[i+3] == 'h':
                list.append('cosh')
                i += 4
            elif i+2 < len(str) and str[i+2] == 's':
                list.append('cos')
                i += 3
            else:
                i += 1
                list.append('c')
        elif str[i] == 't':
            if str[i+3] == 'h':
                list.append('tanh')
                i += 4
            elif str[i+2] == 'n':
                list.append('tan')
                i += 3
            else:
                list.append('#')
                break
        elif str[i] == 'a':
            if str[i:i+5] in ['asinh','acosh','atanh']:
                list.append(str[i:i+5])
                i += 5
            elif str[i:i+4] in ['asin','acos','atan']:
                list.append(str[i:i+4])
                i += 4
            else:
                list.append('#')
                break
        elif str[i] == 'm':
            i += 5
        elif str[i] == 'e':
            if str[i+1] == 'x':
                list.append('exp')
                i += 3
            else:
                list.append('e')
                i += 1
        elif str[i] == 'l':
            list.append('log')
            i += 3
        elif str[i] == 'p':
            list.append('pi')
            i += 2
        elif str[i] == 'D':
            list.append('$')
            list.append('y')
            i += 19
        elif str[i] == 'f':
            i += 4
            list.append('f(x)')
        elif str[i] == '#':
            if str[i+2] == 'N':
                i += 27
            elif str[i+2] == 'D':
                i += 13
            elif str[i+2] == 'f':
                i += 4
            else:
                list.append('#')
                break
        else:
#            print('invalid string')
#            print(str[i])
            list.append('#')
            break
    return list
                


def buildTree(prefix):
    prefix = prefix.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    for i in prefix:
        if i in ['**', '*', '/', '+', '-']:
            if currentTree.key == '':
                currentTree.setRootVal(i)
                currentTree.insertLeft('')
                pStack.push(currentTree)
                currentTree = currentTree.getLeftChild()
            elif currentTree.leftChild != '' and currentTree.rightChild == None:
                currentTree.insertRight(i)
                currentTree = currentTree.getRightChild()
                currentTree.insertLeft('')
                pStack.push(currentTree)
                currentTree = currentTree.getLeftChild()
            else:
                while currentTree.rightChild != None:
                    currentTree = pStack.pop()
                currentTree.insertRight(i)
                currentTree = currentTree.getRightChild()
                currentTree.insertLeft('')
                pStack.push(currentTree)
                currentTree = currentTree.getLeftChild()
        elif i in ['tan', 'cos', 'sin', 'exp', 'log', 'sqrt', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']:
            if currentTree.key == '':
                currentTree.setRootVal(i)
                currentTree.insertRight('')
                pStack.push(currentTree)
                currentTree = currentTree.getRightChild()
            elif currentTree.leftChild != '' and currentTree.rightChild == None:
                currentTree.insertRight(i)
                currentTree = currentTree.getRightChild()
                currentTree.insertRight('')
                pStack.push(currentTree)
                currentTree = currentTree.getRightChild()
            else:
                while currentTree.rightChild != None:
                    currentTree = pStack.pop()
                currentTree.insertRight(i)
                currentTree = pStack.pop()
        else:
            if currentTree.key == '':
                currentTree.setRootVal(i)
                currentTree = pStack.pop()
            elif currentTree.rightChild == None:
                currentTree.insertRight(i)
                currentTree = pStack.pop()
            else:
                while currentTree.rightChild != None:
                    currentTree = pStack.pop()
                currentTree.insertRight(i)
                currentTree = pStack.pop()
    return eTree

    
    
    

def Generate_funtion_binary_c(num_node):
    D = np.zeros([20,20])
    D[:,0] = 1
    for n in range(20):
        for e in range(19):
            if e > 0 and n > 0:
                D[e,n] = D[e-1,n] + D[e+1,n-1]
#    print(D)
    etree = BinaryTree('')
    currentTree = etree
    pos_list = []
    pos_list.append(currentTree)
    leaf_list = []
    e = 1
    n = num_node
    while n > 0:
        K = np.array([D[e - i + 1, n - 1 ]/D[e , n] for i in range(len(pos_list))])
#        print(K)
        pos = np.random.choice(range(len(pos_list)), p = K)
        currentTree = pos_list[pos]
        op = np.random.choice(['*', '/', '+', '-'])
        currentTree.setRootVal(op)
        currentTree.insertLeft('')
        currentTree.insertRight('')
        pos_list[pos] = currentTree.getLeftChild()
        pos_list.insert(pos+1,currentTree.getRightChild())
        for i in range(pos):
            if pos+3 <= len(pos_list):
                leaf_list.append(pos_list[pos+2])
                pos_list.pop(pos+2)
            else:
                leaf_list.append(pos_list[0])
                pos_list.pop(0)
        e = e - pos + 1
        n -= 1
    leaf_list = leaf_list + pos_list
    for tree in leaf_list:
#        num = np.random.choice(['x','1','2','3','4','5','-3','-1','-2','-4','-5'])
        num = np.random.choice(['x','1','2','3','-3','-1','-2'], p = [0.6,0.1,0.06,0.06,0.06,0.06,0.06])
        tree.setRootVal(num)
    index = np.random.choice(range(len(leaf_list)))
    leaf_list[index].setRootVal('c')
    return etree
        
def Generate_funtion_unary_c(num_node):
    D = np.zeros([20,20])
    D[:,0] = 1
    for n in range(20):
        for e in range(19):
            if e > 0 and n > 0:
                D[e,n] = D[e-1,n] + D[e,n-1] + D[e+1,n-1]
#    print(D)
    etree = BinaryTree('')
    currentTree = etree
    pos_list = []
    pos_list.append(currentTree)
    leaf_list = []
    e = 1
    n = num_node
    while n > 0:
        K = np.array([[D[e - i, n - 1 ]/D[e , n],D[e - i + 1, n - 1 ]/D[e , n]] for i in range(len(pos_list))])
        index = np.random.choice(range(2*len(pos_list)), p = K.ravel())
        pos = np.array([[[i,1],[i,2]] for i in range(len(pos_list))]).reshape(2*len(pos_list),2)[index]
#        print(pos)
        currentTree = pos_list[pos[0]]
        if pos[1] == 2:
            op = np.random.choice(['*', '/', '+', '-'])
            currentTree.setRootVal(op)
            currentTree.insertLeft('')
            currentTree.insertRight('')
            pos_list[pos[0]] = currentTree.getLeftChild()
            pos_list.insert(pos[0]+1,currentTree.getRightChild())
            for i in range(pos[0]):
                if pos[0]+3 <= len(pos_list):
                    leaf_list.append(pos_list[pos[0]+2])
                    pos_list.pop(pos[0]+2)
                else:
                    leaf_list.append(pos_list[0])
                    pos_list.pop(0)
            e = e - pos[0] + 1
        elif pos[1] == 1:
            op = np.random.choice(['tan', 'cos', 'sin', 'exp', 'log', 'sqrt'])
#            'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'
#            op = np.random.choice(['tan', 'cos', 'sin', 'exp', 'log', 'sqrt'])
            currentTree.setRootVal(op)
            currentTree.insertRight('')
            pos_list[pos[0]] = currentTree.getRightChild()
            for i in range(pos[0]):
                if pos[0]+2 <= len(pos_list):
                    leaf_list.append(pos_list[pos[0]+1])
                    pos_list.pop(pos[0]+1)
                else:
                    leaf_list.append(pos_list[0])
                    pos_list.pop(0)
            e = e - pos[0]
#        print(op)
        n -= 1
    leaf_list = leaf_list + pos_list
    for tree in leaf_list:
        #        num = np.random.choice(['x','1','2','3','4','5','-3','-1','-2','-4','-5'])
        num = np.random.choice(['x','1','2','3','-3','-1','-2'], p = [0.6,0.1,0.06,0.06,0.06,0.06,0.06])
        tree.setRootVal(num)
    index = np.random.choice(range(len(leaf_list)))
    leaf_list[index].setRootVal('c')
    return etree

def Generate_funtion_c(num_node):
    boo = np.random.choice([0,1])
    if boo == 0:
        tree = Generate_funtion_unary_c(num_node)
    else:
        tree = Generate_funtion_binary_c(num_node)
    return tree


def integexp(expr, integ):
    integ_exp = sp.integrate(expr,x)
#    print('eeeee',integ_exp)
    integ_str = lambdastr(x,integ_exp)
#    print(integ_str)
    integ_list = string_to_list(integ_str)
#    print('lalala',integ_str)
    tgt_seq = infix_to_prefix(integ_list)
    integ.append(tgt_seq)
#    return integ

def diffexp(expr, diff):
    diff_exp = sp.diff(expr,x)
#    print('eeeee',diff_exp)
    diff_str = lambdastr(x,diff_exp)
#    print(diff_str)
    diff_list = string_to_list(diff_str)
#    print('lalala',diff_str)
    tgt_seq = infix_to_prefix(diff_list)
    diff.append(tgt_seq)

def coefsimplify(infix):
    if infix[-1] in ['#','(']:
        return ['0']
    infix = infix[1:-1]
    def movefwd(list):
        level = 0
        have_x = False
        have_c = False
        index = 0
        for i in range(1,len(list)):
            if list[-i] == ')':
                level += 1
            elif list[-i] == '(':
                level -= 1
            elif list[-i] == 'x':
                have_x = True
            elif list[-i] == 'c':
                have_c = True
            elif list[-i] in ['+','-']:
                if level == 0:
                    index = -i
                    break
            else:
                pass
        if index == 0:
            return index, have_x, have_c
        return index+len(list), have_x, have_c
    index_list = []
    have_x_list = []
    have_c_list = []
    index = len(infix)
    index_list.append(index)
    have_x_list.append(False)
    have_c_list.append(False)
    del_index=[]
    while index != 0:
#        print('ss',infix[:index])
        index, have_x, have_c = movefwd(infix[:index])
        index_list.append(index)
        have_x_list.append(have_x)
        have_c_list.append(have_c)
    if sum(have_c_list) != 1:
        return ['0']
    index_list[-1] = -1
    c_expr = infix[index_list[have_c_list.index(True)]+1:index_list[have_c_list.index(True)-1]]
    def movefwd2(list):
        level = 0
        have_x = False
        have_c = False
        index = 0
        for i in range(1,len(list)):
            if list[-i] == ')':
                level += 1
            elif list[-i] == '(':
                level -= 1
            elif list[-i] == 'x':
                have_x = True
            elif list[-i] == 'c':
                have_c = True
            elif list[-i] in ['/','*']:
                if level == 0:
                    index = -i
                    break
            else:
                pass
        if index == 0:
            return index, have_x, have_c
        return index+len(list), have_x, have_c
    index_list_c = []
    have_x_list_c = []
    have_c_list_c = []
    index = len(c_expr)
    index_list_c.append(index)
    have_x_list_c.append(False)
    have_c_list_c.append(False)
    del_index=[]
    while index != 0:
#        print('ss',infix[:index])
        index, have_x, have_c = movefwd2(c_expr[:index])
        index_list_c.append(index)
        have_x_list_c.append(have_x)
        have_c_list_c.append(have_c)
    if sum(have_c_list_c) != 1:
        return ['0']
#    print(c_expr)
#    print(have_c_list_c)
#    print(have_x_list_c)
#    print(index_list_c)
    index_list_c[-1] = -1
    if have_x_list_c[have_c_list_c.index(True)] == False:
        for i in range(1,len(index_list_c)):
            if have_x_list_c[i] == False and have_c_list_c[i] == False:
                c_expr[index_list_c[i]+1:index_list_c[i-1]] = ['1']
            elif have_x_list_c[i] == False and have_c_list_c[i] == True:
                c_expr[index_list_c[i]+1:index_list_c[i-1]] = ['c']
            else:
                pass
    print(c_expr)
    infix[index_list[have_c_list.index(True)]+1:index_list[have_c_list.index(True)-1]] = c_expr
    return infix
        


def delconstant(infix):
    if infix[-1] in ['#','(']:
        return ['0']
    infix = infix[1:-1]
    def movefwd(list):
        level = 0
        have_x = False
        index = 0
        for i in range(1,len(list)):
            if list[-i] == ')':
                level += 1
            elif list[-i] == '(':
                level -= 1
            elif list[-i] == 'x':
                have_x = True
            elif list[-i] in ['+','-']:
                if level == 0:
                    index = -i
                    break
            else:
                pass
        if index == 0:
            return index, have_x
        return index+len(list), have_x
    index_list = []
    have_x_list = []
    index = len(infix)
    index_list.append(index)
    have_x_list.append(True)
    del_index=[]
    while index != 0:
#        print('ss',infix[:index])
        index, have_x = movefwd(infix[:index])
        index_list.append(index)
        have_x_list.append(have_x)
    if False in have_x_list:
        for i in range(len(have_x_list)):
            if have_x_list[i] == False:
                del_index.append(index_list[i])
                del_index.append(index_list[i-1])
    if del_index != []:
        max_i = max(del_index)
        min_i = min(del_index)
        rr= infix[:min_i]+infix[max_i:]
        if rr == []:
            return ['0']
        else:
            return rr
    return infix
    
def clear(list):
    
    return list

def soleq(num_node,sol):
    tree = Generate_funtion_c(num_node)
    res = []
    InorderTree(tree, res)
    exp_list = res
    expr = sp.sympify("".join(exp_list))
    print('src expr',expr)
    expr_list = string_to_list_c(lambdastr(x,expr))
    if expr_list in [['('],['0']] or expr_list[-1] == '#':
        sol.append('#')
        return 0
#    expr_list_clear = coefsimplify(expr_list)
#    print(expr_list_clear)
#    if expr_list_clear in [['('],['0']] or expr_list_clear[-1] == '#':
#        sol.append('#')
#        return 0
#    expr = sp.sympify("".join(expr_list_clear))
    f=sp.Function('f')(x)
    equation1 = sp.Eq(expr,f)
    c_solve = sp.solve(equation1,c)
#    print('sssssssssss',c_solve)
    if c_solve == []:
        sol.append('#')
        return 0
    c_solve_clear = delconstant(string_to_list_c(lambdastr(x,c_solve[0])))
    c_clear = sp.sympify("".join(c_solve_clear))
    equation2 = sp.Eq(c_clear,c)
#    print('eq2',equation2)
    f_solve = sp.solve(equation2,f)
    if f_solve == []:
#        print('uhuhuhuuh')
        sol.append('#')
        return 0
    src = infix_to_prefix(string_to_list_c(lambdastr(x,f_solve[0])))
    if src == 0:
        sol.append('#')
        return 0
    equation3 = c_clear.diff(x)
#    print('eq',equation3)
#    print('ss',eq_str)
    trg = infix_to_prefix(string_to_list_c(lambdastr(x,equation3)))
#    print(eq_list)
    if trg == 0:
        sol.append('#')
        return 0
    sol.append(trg)
    sol.append(src)

def Generate_data_ode(num_node):
    manager = Manager()
    sol = manager.list([' '])
    sol_p = Process(target = soleq, args = [num_node, sol])
    start = time.time()
    sol_p.start()
    while True:
        if sol[-1] == ' ' and (time.time() - start)<10:
            pass
        else:
            os.kill(sol_p.pid,signal.SIGKILL)
            break
    trg = []
    src = []
    if len(sol)!=3:
        return 0
#    print('tgt',sol[1])
    if sol[1] =='#':
        return 0
    for str in sol[1].split():
        if str[0] in ['1','2','3','4','5','6','7','8','9','0'] and len(str) > 1:
            for i in range(len(str)):
                trg.append(str[i])
        elif str[0:2] in ['-1','-2','-3','-4','-5','-6','-7','-8','-9'] and len(str) > 2:
            trg.append(str[0:2])
            for i in range(len(str)-2):
                trg.append(str[i+2])
        else:
            trg.append(str)
    for str in sol[2].split():
        if str[0] in ['1','2','3','4','5','6','7','8','9','0'] and len(str) > 1:
            for i in range(len(str)):
                src.append(str[i])
        elif str[0:2] in ['-1','-2','-3','-4','-5','-6','-7','-8','-9'] and len(str) > 2:
            src.append(str[0:2])
            for i in range(len(str)-2):
                src.append(str[i+2])
        else:
            src.append(str)
    return [trg, src]
    



def Generate_dataset_ode(num_seq, num_node, save_path):
    dataset = []
    for i in tqdm(range(num_seq)):
        data = 0
        while data == 0:
            num = np.random.choice(range(num_node))+2
            data = Generate_data_ode(num)
        dataset.append(data)
    print(dataset)
    np.save(save_path + '.npy',np.array(dataset))
    


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-save_path', required=True)

    parser.add_argument('-num_node', type=int, default=5)
    parser.add_argument('-num_seq', type=int, default=2)
#    parser.add_argument('-method', type=str, choices = ['bwd','fwd'],default = 'bwd')
    
    opt = parser.parse_args()
#    if opt.method == 'bwd':
#        Generate_dataset_bwd(opt.num_seq, opt.num_node, opt.save_path)
#    elif opt.method == 'fwd':
#        Generate_dataset_fwd(opt.num_seq, opt.num_node, opt.save_path)
    Generate_dataset_ode(opt.num_seq, opt.num_node, opt.save_path)
if __name__ == '__main__':
    main()
