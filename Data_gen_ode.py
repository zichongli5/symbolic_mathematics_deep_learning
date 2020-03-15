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
            list.append('y')
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
                
def delconstant(infix_list):
    if infix_list[-1] in ['#','(']:
        return ['#']
    infix = infix_list[1:-1]
    level = 0
    have_x = 0
    delete_index = 0
    for i in range(1,len(infix)+1):
        if infix[-i] == ')':
            level += 1
        elif infix[-i] == '(':
            level -= 1
        elif infix[-i] == 'x':
            have_x = 1
            break
        elif infix[-i] in ['+','-']:
            if level == 0:
                delete_index = -i
        else:
            pass
    if have_x == 0:
        infix = ['0']
        return infix
    elif delete_index != 0:
        return infix[:delete_index]
    else:
        return infix


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


def soleq(eq,sol):
    c_solve = sp.solve(eq,c)
    print('sssssssssss',c_solve)
    if c_solve == []:
        sol.append('#')
        return 0
    equation2 = c_solve[0].diff(x)
    #    print(equation2)
    eq_str = lambdastr(x,equation2)
    print('ss',eq_str)
    eq_list = string_to_list_c(eq_str)
    print(eq_list)
    if eq_list in [['('],['0']] or eq_list[-1] == '#':
        sol.append('#')
        return 0
    tg = infix_to_prefix(eq_list)
    sol.append(tg)

def Generate_data_ode(num_node):
    tree = Generate_funtion_c(num_node)
    res = []
    manager = Manager()
    sol = manager.list([' '])
    InorderTree(tree, res)
    exp_list = res
    exp_str = "".join(exp_list)
    expr = sp.sympify(exp_str)
    expr_str = lambdastr(x,expr)
    print('src',expr)
    expr_list = string_to_list_c(expr_str)
    print(expr_str)
#    print('ssss',expr_list)
    if expr_list in [['('],['0']] or expr_list[-1] == '#':
        return 0
    f=sp.Function('f')(x)
    equation1 = sp.Eq(expr,f)
#    print(equation1)

    sol_p = Process(target = soleq, args = [equation1, sol])
    start = time.time()
    sol_p.start()
    while True:
        if sol[-1] == ' ' and (time.time() - start)<5:
            pass
        else:
            os.kill(sol_p.pid,signal.SIGKILL)
            break
#    if c_solve == []:
#        return 0
#    equation2 = c_solve[0].diff(x)
##    print(equation2)
#    eq_str = lambdastr(x,equation2)
#    print('ss',eq_str)
#    eq_list = string_to_list_c(eq_str)
#    print(eq_list)
#    if eq_list in [['('],['0']] or eq_list[-1] == '#':
#        return 0
##    eq_clear = delconstant(eq_list)
    trg = []
    src = []
    if len(sol)==1:
        return 0
    tgt = sol[1]
    print('tgt',tgt)
    if tgt =='#':
        return 0
    for str in tgt.split():
        if str[0] in ['1','2','3','4','5','6','7','8','9','0'] and len(str) > 1:
            for i in range(len(str)):
                trg.append(str[i])
        elif str[0:2] in ['-1','-2','-3','-4','-5','-6','-7','-8','-9'] and len(str) > 2:
            trg.append(str[0:2])
            for i in range(len(str)-2):
                trg.append(str[i+2])
        else:
            trg.append(str)
    for str in infix_to_prefix(expr_list).split():
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
