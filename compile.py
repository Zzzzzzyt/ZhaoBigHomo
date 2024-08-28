import re

CYELLOW = "\033[93m"
CGREEN = "\033[32m"
CRED = "\033[31m"
CCYAN = "\033[36m"
CGREY = "\033[2m"
CRESET = "\033[0m"

# statement
# SEXP = -1
# SBLOCK = 0
# SVAR = 1
# SDEF = 2
# SRET = 3
# SASSIGN = 4
# SIF = 5
# SIFELSE = 6
# SWHILE = 7
# SINPUT = 8
# SOUTPUT = 9

# expression
ops = {
    "+": "ADD",
    "-": "SUB",
    "*": "MUL",
    "/": "DIV",
    "%": "MOD",
    "=": "EQ",
    ">": "GT",
    "<": "LT",
    "&": "AND",
    "|": "OR",
    "!": "NOT",
}

ADDR_ZERO = 0
ADDR_IMM = 1
ADDR_X = 2
ADDR_Y = 3
ADDR_Z = 4
ADDR_LIT_START = 20
ADDR_RET = 100
ADDR_GLOBAL_START = 101
ADDR_SP = 200
ADDR_STACK_START = 201

"""
0   STI     i   -   addr            store immediate     MEM[addr]=i
1   STR     x   B   idx             store relative      MEM[B+MEM[idx]]=MEM[x]
2   LDR     B   idx x               load relative       MEM[x]=MEM[B+MEM[idx]]
3   MOV     x   -   y               move                MEM[y]=MEM[x]

4-14 op     x   y   z               *                   MEM[z]=MEM[x]+MEM[y]
4   ADD     x   y   z               add                 
5   SUB                             subtract
6   MUL                             multiply
7   DIV                             divide
8   MOD                             modulo
9   EQ                              equal
10  GT                              greater than
11  LT                              less than
12  AND                             logical and
13  OR                              logical or
14  NOT     x   -   z               logical not         MEM[z]=!MEM[x]

20  JNZ     x   -   B               jump not zero       if(MEM[x]) PC=B
30  JMP     -   -   B               jump nonconditional PC=B
40  JPI     -   -   x               jump indirect       PC=MEM[x]
50  OUT     x   -   -               output              print(MEM[x])
60  IN      x   -   -               input               MEM[x]=input()
"""

opcodes = {
    "STI": (0, [0, 2]),
    "STR": (1, [0, 1, 2]),
    "LDR": (2, [0, 1, 2]),
    "MOV": (3, [0, 2]),
    "ADD": (4, [0, 1, 2]),
    "SUB": (5, [0, 1, 2]),
    "MUL": (6, [0, 1, 2]),
    "DIV": (7, [0, 1, 2]),
    "MOD": (8, [0, 1, 2]),
    "EQ": (9, [0, 1, 2]),
    "GT": (10, [0, 1, 2]),
    "LT": (11, [0, 1, 2]),
    "AND": (12, [0, 1, 2]),
    "OR": (13, [0, 1, 2]),
    "NOT": (14, [0, 2]),
    "JNZ": (20, [0, 2]),
    "JMP": (30, [2]),
    "JPI": (40, [2]),
    "OUT": (50, [0]),
    "IN": (60, [0]),
}

"""
stack frame layout
MEM[ADDR_SP]=frame_base

base+0  ret_addr
    +1  local_var1
    +2  local_var2

frame_size=3



call:
MEM[ADDR_SP]+=current_frame_size
MEM[fb+0]=PC
JMP
MEM[ADDR_SP]-=current_frame_size
"""


def preprocess(s):
    s = s.split("\n")
    s = list(map(str.strip, s))
    s = "".join(s)
    s = s.replace("\t", "")
    s = s.replace("\n", "")
    s = re.sub(r"\ +", " ", s)
    s = s.replace("{", ";{")
    s = s.replace("}", "};")
    s = s.replace("&&", "＆")
    s = s.replace("||", "｜")
    s = s.replace("==", "＝")
    s = s.replace("if", "if;")
    s = s.replace("else", "else;")
    s = s.replace("while", "while;")
    s = re.sub(r"(def\ \w*\(\))", lambda x: x.group(0) + ";", s)
    return s


src = open("src.zbh").read()
src = preprocess(src)
print(src)


def tokenize(s: str):
    tokens = []
    i = 0
    t = "op"
    cur = ""
    while i < len(s):
        c = s[i]
        if c in "+-*/%><!｜＆&＝)," or (c in "(" and cur == ""):
            if len(cur):
                tokens.append((t, cur))
            if c == "＝":
                c = "=="
            elif c == "＆":
                c = "&&"
            elif c == "｜":
                c = "||"
            tokens.append(("op", c))
            t = "op"
            cur = ""
        elif c in "(":
            tokens.append(("op", "call#" + cur))
            tokens.append(("op", "("))
            cur = ""
            t = "op"
        elif c.isdigit():
            t = "lit"
            cur += c
        else:
            t = "var"
            cur += c
        i += 1
    if len(cur):
        tokens.append((t, cur))
    for i in range(len(tokens)):
        typ, val = tokens[i]
        if typ == "op":
            if val == "(":
                tokens[i] = ("lp", val)
            elif val == ")":
                tokens[i] = ("rp", val)
            elif val in "+-!*&" and (i == 0 or tokens[i - 1][0] == "op"):
                tokens[i] = ("op1", "_" + val)
            else:
                tokens[i] = ("op2", val)
    return tokens


litvals = {0: 0}


def newlit(val):
    if val not in litvals:
        litvals[val] = len(litvals) + ADDR_LIT_START
    return litvals[val]


priority = {
    "(": -100,
    "_*": 9,
    "_&": 9,
    "_!": 8,
    "_+": 8,
    "_-": 8,
    "*": 7,
    "/": 7,
    "%": 7,
    "+": 6,
    "-": 6,
    ">": 5,
    "<": 5,
    "==": 4,
    "!": 3,
    "&&": 2,
    "||": 1,
    ",": 0,
}


def genexp(s: str, depth=0):
    pad = "  " * depth
    print(pad, CGREEN + "genexp:" + CRESET)
    print(pad, CGREY + s + CRESET)
    # =a+b
    #  ^ ^
    # (a+b)
    #  ^ ^

    # ()   call   */%    +-  ><   ==   !   &&  ||
    tokens = tokenize(s)
    print(pad, tokens)

    rt = []
    op = []

    def proc():
        o = op.pop()
        if o.startswith("call#"):
            func = o[5:]
            args = []
            x = rt.pop()
            while x[0] == ",":
                args.append(x[1])
                x = x[2]
            args.append(x)
            rt.append(("call", func, args))
        elif o[0] == "_":
            x = rt.pop()
            rt.append((o, x))
        else:
            x = rt.pop()
            y = rt.pop()
            rt.append((o, y, x))

    for typ, val in tokens:
        if typ == "lit":
            val = int(val)
            newlit(val)
            rt.append(("lit", int(val)))
        elif typ == "var":
            rt.append(("var", val))
        else:
            if val == "(":
                op.append("(")
            elif val == ")":
                while op[-1] != "(":
                    proc()
                op.pop()
                if len(op) and op[-1].startswith("call#"):
                    print("zjs")
                    name = op[-1][5:]
                    rt.append(("call", name, []))
                    op.pop()
            else:
                if val.startswith("call#"):
                    rassoc = True
                    p1 = 9
                else:
                    rassoc = len(val) == 2
                    p1 = priority[val]
                while True:
                    if len(op) == 0:
                        break
                    val2 = op[-1]
                    if val2.startswith("call#"):
                        p2 = 9
                    else:
                        p2 = priority[val2]
                    if rassoc:
                        if p2 <= p1:
                            break
                    else:
                        if p2 < p1:
                            break
                    proc()
                op.append(val)
    while len(op):
        proc()
    print(rt)
    assert len(rt) == 1
    return rt[0]


funcargs = {}


def genstat(s: str, depth=0):
    # {aaa;bbb;{ccc};}
    #  ^            ^
    pad = "  " * depth
    print(pad, CGREEN + "genstat:" + CRESET)
    print(pad, CGREY + s + CRESET)
    i = 0
    rt = []
    flag = False
    while True:
        cur = ""
        exp = None
        while True:
            if i >= len(s):
                break
            # print(c, s[c])
            if s[i] == "(" or (cur == "" and (s[i].isdigit() or s[i] in "+-()")):
                if cur.startswith("def "):
                    j = s.find(")", i)
                    args = s[i + 1 : j].split(",")
                    args = list(filter(lambda x: len(x) > 0, args))
                    i = j + 1
                    name = cur[4:]
                    rt.append(("SDEF", name, args))
                    cur = ""
                    funcargs[name] = args
                    continue
                elif cur.startswith("output ") or cur.startswith("return "):
                    pass
                else:
                    nxt = s.find(";", i)
                    if cur != "":
                        if "=" in s[i:nxt]:
                            cur += s[i]
                            i += 1
                            continue
                        # f()
                        exp = genexp(cur + s[i:nxt], depth=depth + 1)
                        rt.append(("SEXP", exp))
                    else:
                        # (11111)
                        exp = genexp(s[i:nxt], depth=depth + 1)
                        rt.append(("SEXP", exp))
                    i = nxt + 1
                    cur = ""
                    continue
            elif s[i] == "=":
                nxt = s.find(";", i)
                exp = genexp(s[i + 1 : nxt], depth=depth + 1)
                if cur.startswith("var "):
                    rt.append(("SVAR", cur[4:]))
                    rt.append(("SASSIGN", cur[4:], exp))
                else:
                    if cur[0] == "*":
                        rt.append(("SASSIGNP", genexp(cur[1:], depth=depth + 1), exp))
                    else:
                        rt.append(("SASSIGN", cur, exp))
                i = nxt + 1
                cur = ""
                continue
            elif s[i] == "{":
                tmp, l = genstat(s[i + 1 :], depth=depth + 1)
                rt.append(("SBLOCK", tmp))
                i += l + 3
                cur = ""
                continue
            elif s[i] == ";":
                if cur != "":
                    rt.append((-1, cur))
                i += 1
                cur = ""
                continue
            elif s[i] == "}":
                flag = True
                break
            cur += s[i]
            i += 1
        # print("  " * depth, rt, exp)
        if flag:
            break
        if i >= len(s):
            break
    rt2 = []
    j = 0
    while j < len(rt):
        r = rt[j]
        t = r[0]
        if t == "SBLOCK":
            rt2.extend(r[1])
        elif t == "SASSIGN" or t == "SASSIGNP" or t == "SVAR" or t == "SEXP":
            rt2.append(r)
        elif t == "SDEF":
            bl = rt[j + 1]
            assert bl[0] == "SBLOCK"
            rt2.append(("SDEF", r[1], r[2], bl[1]))
            j += 1
        else:
            cur = r[1]
            if cur[:4] == "var ":
                rt2.append(("SVAR", cur[4:]))
            elif cur[:6] == "input ":
                rt2.append(("SINPUT", cur[6:]))
            elif cur[:7] == "output ":
                rt2.append(("SOUTPUT", genexp(cur[7:], depth=depth + 1)))
            elif cur[:6] == "return":
                if len(cur) == 6:
                    rt2.append(("SRETURN", None))
                else:
                    rt2.append(("SRETURN", genexp(cur[7:], depth=depth + 1)))
            elif cur == "while":
                exp = rt[j + 1]
                bl = rt[j + 2]
                assert exp[0] == "SEXP"
                assert bl[0] == "SBLOCK"
                rt2.append(("SWHILE", exp[1], bl[1]))
                j += 2
            elif cur == "if":
                exp = rt[j + 1]
                bl1 = rt[j + 2]
                assert exp[0] == "SEXP"
                assert bl1[0] == "SBLOCK"
                bl2 = None
                if j + 3 < len(rt):
                    if rt[j + 3][1].startswith("else"):
                        bl2 = rt[j + 4]
                        assert bl2[0] == "SBLOCK"
                if bl2:
                    rt2.append(("SIFELSE", exp[1], bl1[1], bl2[1]))
                    j += 4
                else:
                    rt2.append(("SIF", exp[1], bl1[1]))
                    j += 2

        j += 1
    print(pad, "len:", i)
    # print(pad, rt)
    print(pad, rt2)
    print()
    return rt2, i


ast = genstat(src)[0]

globalvar = {}

for i in ast:
    if i[0] == "SVAR":
        name = i[1]
        globalvar[name] = ADDR_GLOBAL_START + len(globalvar)

print("GLOBAL VARS", globalvar)


def exec_exp(exp, localvar, base, depth=0):
    pad = "  " * depth
    print(pad, CGREEN + "exec_exp:" + CRESET, "base =", base)
    print(pad, CGREY, exp, CRESET)

    t = exp[0]
    if t == "lit":
        adr = newlit(exp[1])
        # return [("STI", exp[1], ADDR_IMM), ("mov dr", ADDR_IMM, base)], 1
        return [("mov dr", adr, base)], 1
    elif t == "var":
        name = exp[1]
        if name not in localvar:
            return [("mov dr", globalvar[name], base)], 1
        else:
            return [("mov rr", localvar[name], base)], 1
    elif t == "call":
        func = exp[1]
        args = exp[2]
        mem = len(args)
        rt = []
        for i in range(len(args)):
            tmp1, mem1 = exec_exp(args[i], localvar, base + i, depth + 1)
            mem = max(mem, i + mem1)
            rt.extend(tmp1)
        return rt + [("call l", func, base), ("mov dr", ADDR_RET, base)], max(mem, 1)
    elif t == "_+" or t == "_-":
        tmp, mem = exec_exp(exp[1], localvar, base, depth + 1)
        if t == "_-":
            tmp += [("op drr", t[0], ADDR_ZERO, base, base)]
        return tmp, mem
    elif t == "_!":
        tmp, mem = exec_exp(exp[1], localvar, base, depth + 1)
        return tmp + [("not rr", base, base)], mem
    elif t == "_*":
        tmp, mem = exec_exp(exp[1], localvar, base, depth + 1)
        tmp.append(("LDR", base, ADDR_SP, ADDR_X))
        tmp.append(("LDR", 0, ADDR_X, ADDR_X))
        tmp.append((("STR", ADDR_X, base, ADDR_SP)))
        return tmp, mem
    elif t == "_&":
        assert exp[1][0] == "var"
        name = exp[1][1]
        if name not in localvar:
            tmp = [("STI", globalvar[name], ADDR_X), ("mov dr", ADDR_X, base)]
        else:
            adr = newlit(localvar[name])
            tmp = [("ADD", ADDR_SP, adr, ADDR_X), ("mov dr", ADDR_X, base)]
        return tmp, 1
    else:
        tmp1, mem1 = exec_exp(exp[1], localvar, base, depth + 1)
        tmp2, mem2 = exec_exp(exp[2], localvar, base + 1, depth + 1)
        return tmp1 + tmp2 + [("op rrr", t, base, base + 1, base)], max(mem1, mem2) + 1


labels = {}


def new_label():
    name = "#label_" + str(len(labels))
    labels[name] = None
    return name


def exec_block(block, localvar0, base, depth=0):
    pad = "  " * depth
    print(pad, CGREEN + "exec_block:" + CRESET, "base =", base)
    print(pad, CGREY, block, CRESET)
    print()

    mem = 0
    localvar = localvar0.copy()
    localmem = 0
    expbase = base
    rt = []
    rt.append(("comment", "start of block:" + str(block)))
    for stat in block:
        t = stat[0]
        rt.append(("comment", stat))
        if t == "SEXP":
            tmp, tmpmem = exec_exp(stat[1], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem)
            rt.extend(tmp)
        elif t == "SBLOCK":
            tmp, tmpmem = exec_block(stat[1], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem)
            rt.extend(tmp)
        elif t == "SVAR":
            localvar[stat[1]] = expbase
            expbase += 1
            localmem += 1
            mem = max(mem, localmem)
        elif t == "SDEF":
            # WTF??????????
            print(CRED + "non global functions are not supported yet!!!" + CRESET)
        elif t == "SRETURN":
            if stat[1]:
                if stat[1][0] == "var":
                    if stat[1][1] in localvar:
                        rt.append(("mov rd", localvar[stat[1][1]], ADDR_RET))
                    else:
                        rt.append(("mov dd", globalvar[stat[1][1]], ADDR_RET))
                elif stat[1][0] == "lit":
                    adr = newlit(stat[1][1])
                    rt.append(("mov dd", adr, ADDR_RET))
                else:
                    tmp, tmpmem = exec_exp(stat[1], localvar, expbase, depth + 1)
                    mem = max(mem, localmem + tmpmem)
                    rt.extend(tmp)
                    rt.append(("mov rd", expbase, ADDR_RET))
            rt.append(("jmp r", 0))
        elif t == "SASSIGN":
            name = stat[1]
            exp = stat[2]
            if exp[0] == "var":
                name2 = exp[1]
                if name2 not in localvar:
                    if name not in localvar:
                        rt.append(("mov dd", globalvar[name2], globalvar[name]))
                    else:
                        rt.append(("mov dr", globalvar[name2], localvar[name]))
                else:
                    if name not in localvar:
                        rt.append(("mov rd", localvar[name2], globalvar[name]))
                    else:
                        rt.append(("mov rr", localvar[name2], localvar[name]))
            elif exp[0] == "lit":
                val = exp[1]
                adr = newlit(val)
                if name not in localvar:
                    rt.append(("mov dd", adr, globalvar[name]))
                else:
                    rt.append(("mov dr", adr, localvar[name]))
            else:
                tmp, tmpmem = exec_exp(exp, localvar, expbase, depth + 1)
                mem = max(mem, localmem + tmpmem)
                rt.extend(tmp)
                if name not in localvar:
                    rt.append(("mov rd", expbase, globalvar[name]))
                else:
                    rt.append(("mov rr", expbase, localvar[name]))
        elif t == "SASSIGNP":
            exp1 = stat[1]
            exp2 = stat[2]
            tmp1, tmpmem1 = exec_exp(exp1, localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem1)
            tmp2, tmpmem2 = exec_exp(exp2, localvar, expbase + 1, depth + 1)
            mem = max(mem, localmem + 1 + tmpmem2)
            rt.extend(tmp1)
            rt.extend(tmp2)
            rt.append(("LDR", expbase, ADDR_SP, ADDR_X))
            rt.append(("LDR", expbase + 1, ADDR_SP, ADDR_Y))
            rt.append(("STR", ADDR_Y, 0, ADDR_X))
        elif t == "SIF":
            tmp, tmpmem = exec_exp(stat[1], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem)
            rt.extend(tmp)

            label = new_label()
            rt.append(("op rrr", "!", expbase, None, expbase))
            rt.append(("jnz rl", expbase, label))

            tmp2, tmpmem2 = exec_block(stat[2], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem2)
            rt.extend(tmp2)

            rt.append(("label", label))
        elif t == "SIFELSE":
            # exp
            # else block
            # label1
            # if block
            # label2

            tmp, tmpmem = exec_exp(stat[1], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem)
            rt.extend(tmp)

            label1 = new_label()
            label2 = new_label()

            tmp2, tmpmem2 = exec_block(stat[2], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem2)
            tmp3, tmpmem3 = exec_block(stat[3], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem3)

            rt.append(("jnz rl", expbase, label1))
            rt.extend(tmp3)
            rt.append(("jmp l", label2))
            rt.append(("label", label1))
            rt.extend(tmp2)
            rt.append(("label", label2))

        elif t == "SWHILE":
            label1 = new_label()
            label2 = new_label()

            rt.append(("label", label1))

            tmp, tmpmem = exec_exp(stat[1], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem)
            rt.extend(tmp)

            rt.append(("op rrr", "!", expbase, None, expbase))
            rt.append(("jnz rl", expbase, label2))

            tmp2, tmpmem2 = exec_block(stat[2], localvar, expbase, depth + 1)
            mem = max(mem, localmem + tmpmem2)
            rt.extend(tmp2)

            rt.append(("jmp l", label1))
            rt.append(("label", label2))
        elif t == "SINPUT":
            name = stat[1]
            if name not in localvar:
                rt.append(("in d", globalvar[name]))
            else:
                rt.append(("in r", localvar[name]))
        elif t == "SOUTPUT":
            if stat[1][0] == "var":
                if stat[1][1] in localvar:
                    rt.append(("out r", localvar[stat[1][1]]))
                else:
                    rt.append(("out d", globalvar[stat[1][1]]))
            elif stat[1][0] == "lit":
                adr = newlit(stat[1][1])
                rt.append(("out d", adr))
            else:
                tmp, tmpmem = exec_exp(stat[1], localvar, expbase, depth + 1)
                mem = max(mem, localmem + tmpmem)
                rt.extend(tmp)
                rt.append(("out r", expbase))
    rt.append(("comment", "end of block"))
    return rt, mem


def exec_func(func):
    print(CGREEN + "exec_func" + CRESET)
    print(CGREY, func, CRESET)
    print()
    name = func[1]
    args = func[2]
    localvar = {}
    expbase = 1
    for i in args:
        localvar[i] = expbase
        expbase += 1
    block, mem = exec_block(func[3], localvar, expbase)
    labels[name] = None
    block.insert(0, ("label", name))
    return block, mem


def gen_asm(asm):
    # labelpos = {}
    rt = []

    def gen(*args):
        rt.append(args)

    curfunc = None
    for i in asm:
        if i[0] == "comment":
            gen(i[0], i[1])
            continue
        if i[0] == "label":
            name = i[1]
            # labelpos[name] = cnt
            if not name.startswith("#"):
                curfunc = name
            gen("label", name)
            continue
        if i[0][0].isupper():
            gen(*i)
            continue
        t, modes = i[0].split()
        args = list(i[1:])
        if t == "mov":
            if modes == "dd":
                gen("MOV", args[0], args[1])
            elif modes == "dr":
                gen("STR", args[0], args[1], ADDR_SP)
            elif modes == "rd":
                gen("LDR", args[0], ADDR_SP, args[1])
            else:
                gen("LDR", args[0], ADDR_SP, ADDR_X)
                gen("STR", ADDR_X, args[1], ADDR_SP)
        elif t == "not":
            if modes[0] == "r":
                gen("LDR", args[0], ADDR_SP, ADDR_X)
                x = ADDR_X
            else:
                x = args[0]
            if modes[1] == "r":
                gen("NOT", x, x)
                gen("STR", x, args[1], ADDR_SP)
            else:
                gen("NOT", x, args[1])
        elif t == "op":
            if modes[0] == "r":
                gen("LDR", args[1], ADDR_SP, ADDR_X)
                x = ADDR_X
            else:
                x = args[1]
            if args[2]:
                if modes[1] == "r":
                    gen("LDR", args[2], ADDR_SP, ADDR_Y)
                    y = ADDR_Y
                else:
                    y = args[2]
                if modes[2] == "r":
                    gen(ops[args[0]], x, y, x)
                    gen("STR", x, args[3], ADDR_SP)
                else:
                    gen(ops[args[0]], x, y, args[3])
            else:
                if modes[2] == "r":
                    gen(ops[args[0]], x, x)
                    gen("STR", x, args[3], ADDR_SP)
                else:
                    gen(ops[args[0]], x, args[3])
        elif t == "in":
            if modes[0] == "r":
                gen("IN", ADDR_X)
                gen("STR", ADDR_X, args[0], ADDR_SP)
            else:
                gen("IN", args[0])
        elif t == "out":
            if modes[0] == "r":
                gen("LDR", args[0], ADDR_SP, ADDR_X)
                gen("OUT", ADDR_X)
            else:
                gen("OUT", args[0])
        elif t == "jmp":
            if modes[0] == "r":
                gen("LDR", args[0], ADDR_SP, ADDR_X)
                gen("JPI", ADDR_X)
            else:
                gen(*i)
        elif t == "jnz":
            if modes[0] == "r":
                gen("LDR", i[1], ADDR_SP, ADDR_X)
                gen("jnz dl", ADDR_X, i[2])
                # LDR args[0] ADDR_SP ADDR_X
                # JNZ ADDR_X label
            else:
                gen(*i)
        elif t == "call":
            # STI addr ADDR_IMM
            # STI cur_func_mem ADDR_X
            # ADD ADDR_SP ADDR_X ADDR_SP
            # STR ADDR_IMM 1 ADDR_SP
            # JMP label_pos
            # STI cur_func_mem ADDR_X
            # SUB ADDR_SP ADDR_X ADDR_SP
            curmem = funcmem[curfunc]
            ret_adr = newlit(curmem)
            label = new_label()
            callname = args[0]
            base = args[1]
            for i in range(len(funcargs[callname])):
                adr1 = base + i
                adr2 = curmem + 1 + i
                gen("LDR", adr1, ADDR_SP, ADDR_X)
                gen("STR", ADDR_X, adr2, ADDR_SP)
            gen("ADD", ADDR_SP, ret_adr, ADDR_SP)
            gen("str l", label, 0, ADDR_SP)
            gen("jmp l", args[0])
            gen("label", label)
            gen("SUB", ADDR_SP, ret_adr, ADDR_SP)
            # gen(*i)
            # rt.append(("call", args[0]))
        else:
            gen(*i)
    gen("STI", ADDR_STACK_START, ADDR_SP)
    gen("STI", ADDR_STACK_START, None)
    gen("jmp l", "main")
    return rt


def calc_label(rt):
    rt2 = []
    labelpos = {}
    cnt = 0
    for idx in range(len(rt)):
        i = rt[idx]
        t = i[0]
        if t == "comment":
            continue
        if t == "label":
            labelpos[i[1]] = cnt
            continue
        cnt += 1

    for idx in range(len(rt)):
        i = rt[idx]
        t = i[0]
        if t == "label" or t == "comment":
            continue
        elif t == "jmp l":
            rt2.append(("JMP", labelpos[i[1]]))
        elif t == "jnz dl":
            rt2.append(("JNZ", i[1], labelpos[i[2]]))
        elif t == "str l":
            adr = newlit(labelpos[i[1]])
            rt2.append(("STR", adr, i[2], i[3]))
        else:
            rt2.append(i)
    return rt2


def print_asm(asm):
    s = ""
    for i in asm:
        # cnt = i[0]
        t = i[0]
        args = list(i[1:])
        # s += str(cnt).ljust(6)
        s += t.ljust(10)
        s += ", ".join(map(str, list(args))) + "\n"
        # print(args)
    return s


funcmem = {}
funcsz = {}
ir = []
for i in ast:
    if i[0] == "SDEF":
        name = i[1]
        block, mem = exec_func(i)
        print(CYELLOW + "function generated:", name, mem, CRESET)
        funcmem[name] = mem
        ir.extend(block)

open("src.zbi", "w").write(print_asm(ir))

asm = gen_asm(ir)
asm = [("JMP", None)] + asm

open("src.zbs", "w").write(print_asm(asm))

asm = calc_label(asm)
asm[0] = ("JMP", len(asm) - 3)
init = []
for val, adr in litvals.items():
    init.append(("STI", val, adr))
asm = asm[:-3] + init + asm[-3:]
asm[-2] = ("STI", len(asm), ADDR_STACK_START)


def assemble(asm):
    s = ""
    for i in asm:
        opcode, pos = opcodes[i[0]]
        args = list(i[1:])
        l = [""] * 3
        for i in range(len(pos)):
            l[pos[i]] = str(args[i])
        s += "\t".join([str(opcode)] + l) + "\n"
    return s


bin = assemble(asm)
print(bin)
open("src.zbb", "w").write(bin)
