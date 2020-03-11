from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
import numpy as np
import sympy as sp
from sympy.abc import x,y
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

    prec = {')': 4, 'tan': 4, 'cos': 4, 'sin': 4, '**': 4, '*': 3, '/': 3, '+': 2, '-': 2, '(': 1, 'exp':4, 'log':4, 'sqrt':4, 'asin':4, 'acos':4, 'atan':4, 'sinh':4, 'cosh':4, 'tanh':4, 'asinh':4, 'acosh':4, 'atanh':4}

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



def string_to_list(str):
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
#                print('invalid string')
#                print(str[i])
                break
        elif str[i] == 'c':
            if str[i+3] == 'h':
                list.append('cosh')
                i += 4
            elif str[i+2] == 's':
                list.append('cos')
                i += 3
            else:
#                print('invalid string')
#                print(str[i])
                break
        elif str[i] == 't':
            if str[i+3] == 'h':
                list.append('tanh')
                i += 4
            elif str[i+2] == 'n':
                list.append('tan')
                i += 3
            else:
#                print('invalid string')
#                print(str[i])
                break
        elif str[i] == 'a':
            if str[i:i+5] in ['asinh','acosh','atanh']:
                list.append(str[i:i+5])
                i += 5
            elif str[i:i+4] in ['asin','acos','atan']:
                list.append(str[i:i+4])
                i += 4
            else:
#                print('invalid string')
#                print(str[i])
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
    
def delconstant2(infix):
    if infix[-1] in ['#','(']:
        return ['#']
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

    
    
    
def Generate_funtion_binary(num_node):
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
#    index = np.random.choice(range(len(leaf_list)))
#    leaf_list[index].setRootVal('x')
    return etree
        
def Generate_funtion_unary(num_node):
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
            op = np.random.choice(['tan', 'cos', 'sin', 'exp', 'log', 'sqrt', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'])
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
#    index = np.random.choice(range(len(leaf_list)))
#    leaf_list[index].setRootVal('x')
#    index = np.random.choice(range(len(leaf_list)))
    return etree

def Generate_funtion(num_node):
    boo = np.random.choice([0,1])
    if boo == 0:
        tree = Generate_funtion_unary(num_node)
    else:
        tree = Generate_funtion_binary(num_node)
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


def Generate_data_bwd(num_node):
#    integ_list = ['#']
    manager = Manager()
    diff = manager.list([' '])
    while diff[-1] in [' ', 0]:
        start = time.time()
        diff = manager.list([' '])
#        print('s')
        tree = Generate_funtion(num_node)
        res = []
        InorderTree(tree, res)
        exp_list = res
        exp_str = "".join(exp_list)
#        exp_str = '324'
#        print(exp_str)
        expr = sp.expand(sp.sympify(exp_str))
        expr_str = lambdastr(x,expr)
#        print('src',expr_str)
        expr_list = string_to_list(expr_str)
#        print(expr_list)
        expr_clear = delconstant2(expr_list)
#        print(expr_clear)
        if expr_clear[-1] != '#':
            expr = sp.sympify("".join(expr_clear))
#            print('src',expr)
#        print(expr_clear)
        src_seq = infix_to_prefix(expr_clear)
        diff_p = Process(target = diffexp, args = (expr, diff))
        if src_seq != 0:
#            print('start int.....')
            diff_p.start()
#            integ = integexp(expr)
            while True:
                if diff[-1] == ' ' and (time.time() - start)<5:
                    pass
                else:
#                    print('dd',diff)
                    os.kill(diff_p.pid,signal.SIGKILL)
                    break
#            if integ == None:
#                continue
#            else:
#                integ_str = lambdastr(x,integ)
#                integ_list = string_to_list(integ_str)
#                tgt_seq = infix_to_prefix(integ_list)
        else:
            continue
#    print('src',expr)
#    print('tgt',integ[1])
    trg = []
    src = []
    tgt = diff[1].split()
    for str in tgt:
        if str[0] in ['1','2','3','4','5','6','7','8','9','0'] and len(str) > 1:
            for i in range(len(str)):
                trg.append(str[i])
        elif str[0:2] in ['-1','-2','-3','-4','-5','-6','-7','-8','-9'] and len(str) > 2:
            trg.append(str[0:2])
            for i in range(len(str)-2):
                trg.append(str[i+2])
        else:
            trg.append(str)
    for str in src_seq.split():
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
    
#def Generate_data_ibp(num_node,data):
#    F_tree = Generate_funtion(num_node)
#    G_tree = Generate_funtion(num_node)
#    res1 = []
#    InorderTree(F_tree, res1)
#    res2 = []
#    InorderTree(G_tree, res2)
#    F = sp.sympify("".join(res1))
#    G = sp.sympify("".join(res2))
#    F_list = string_to_list(lambdastr(x,F))
#    G_list = string_to_list(lambdastr(x,G))
#    if F_list[-1] == '#' || G_list[-1] == '#':
#        break
#    f = sp.diff(F,x)
#    g = sp.diff(G,x)
#    Fg = F * g
#    fG = f * G
#    Fg_list = string_to_list(Fg)
#    fG_list = string_to_list(fG)
#    if Fg_list[-1] == '#' || fG_list[-1] == '#':
#        break
#    if Fg_list not in data && fG_list not in data:
#        break
#    data.index(Fg_list)
    
    

def Generate_data_fwd(num_node):
    #    integ_list = ['#']
    manager = Manager()
    integ = manager.list([' '])
    while integ[-1] in [' ', 0]:
        start = time.time()
        integ = manager.list([' '])
#           print('s')
        tree = Generate_funtion(num_node)
        res = []
        InorderTree(tree, res)
        exp_list = res
        exp_str = "".join(exp_list)
    #       exp_str = '324'
#           print(exp_str)
        expr = sp.sympify(exp_str)
        expr_str = lambdastr(x,expr)
        print('src',expr)
        expr_list = string_to_list(expr_str)
        src_seq = infix_to_prefix(expr_list)
        integ_p = Process(target = integexp, args = (expr, integ))
        if src_seq != 0:
#               print('start int.....')
            integ_p.start()
    #           integ = integexp(expr)
            while True:
                if integ[-1] == ' ' and (time.time() - start)<5:
                    pass
                else:
#                       print('dd',integ)
                    os.kill(integ_p.pid,signal.SIGKILL)
                    break
    #            ifinteg == None:
    #               continue
    #            lse:
    #               integ_str = lambdastr(x,integ)
    #               integ_list = string_to_list(integ_str)
    #               tgt_seq = infix_to_prefix(integ_list)
        else:
            continue
#        print('src',expr)
    #    print('tgt',integ[1])
    trg = []
    src = []
    tgt = integ[1].split()
    for str in tgt:
        if str[0] in ['1','2','3','4','5','6','7','8','9','0'] and len(str) > 1:
            for i in range(len(str)):
                trg.append(str[i])
        elif str[0:2] in ['-1','-2','-3','-4','-5','-6','-7','-8','-9'] and len(str) > 2:
            trg.append(str[0:2])
            for i in range(len(str)-2):
                    trg.append(str[i+2])
        else:
            trg.append(str)
    for str in src_seq.split():
        if str[0] in ['1','2','3','4','5','6','7','8','9','0'] and len(str) > 1:
            for i in range(len(str)):
                src.append(str[i])
        elif str[0:2] in ['-1','-2','-3','-4','-5','-6','-7','-8','-9'] and len(str) > 2:
            src.append(str[0:2])
            for i in range(len(str)-2):
                src.append(str[i+2])
        else:
            src.append(str)
    return [src, trg]
    

def Generate_dataset_bwd(num_seq, num_node, save_path):
    dataset = []
    num = num_node+2
    for i in tqdm(range(num_seq)):
        data = Generate_data_bwd(num)
        dataset.append(data)
    print(dataset)
    np.save(save_path + '.npy',np.array(dataset))
    
def Generate_dataset_fwd(num_seq, num_node, save_path):
    dataset = []
    for i in tqdm(range(num_seq)):
        num = np.random.choice(range(num_node))+2
        data = Generate_data_fwd(num)
        dataset.append(data)
    print(dataset)
    np.save(save_path + '.npy',np.array(dataset))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-save_path', required=True)

    parser.add_argument('-num_node', type=int, default=5)
    parser.add_argument('-num_seq', type=int, default=2)
    parser.add_argument('-method', type=str, choices = ['bwd','fwd'],default = 'bwd')
    
    opt = parser.parse_args()
    if opt.method == 'bwd':
        Generate_dataset_bwd(opt.num_seq, opt.num_node, opt.save_path)
    elif opt.method == 'fwd':
        Generate_dataset_fwd(opt.num_seq, opt.num_node, opt.save_path)
    
if __name__ == '__main__':
    main()
