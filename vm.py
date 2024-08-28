prog = open("src.zbb").readlines()
prog = list(map(lambda x: list(map(lambda y: int(y) if len(y) else None, x[:-1].split("\t"))), prog))

MEM = [-7512] * 1000000
PC = 0

memos = {
    0: "STI",
    1: "STR",
    2: "LDR",
    3: "MOV",
    4: "ADD",
    5: "SUB",
    6: "MUL",
    7: "DIV",
    8: "MOD",
    9: "EQ",
    10: "GT",
    11: "LT",
    12: "AND",
    13: "OR",
    14: "NOT",
    20: "JNZ",
    30: "JMP",
    40: "JPI",
    50: "OUT",
    60: "IN",
}

t = 0

debug = False

while True:
    if PC >= len(prog):
        print("end of program reached. halt")
        break
    op = prog[PC]
    opcode = op[0]
    print(str(PC).ljust(6), memos[opcode].ljust(5), ", ".join(map(str, filter(lambda x: x is not None, op[1:]))))

    if debug:
        while True:
            s = input().strip()
            if len(s) == 0:
                break
            else:
                s = s.split()
                if s[0] == "memr" or s[0] == "mem":
                    adr = int(s[1])
                    if s[0] == "memr":
                        adr += MEM[200]
                    if len(s) == 3:
                        MEM[adr] = int(s[2])
                    print(f"MEM[{adr}]={MEM[adr]}")

    t += 1
    if opcode == 0:
        MEM[op[3]] = op[1]
    elif opcode == 1:
        MEM[op[2] + MEM[op[3]]] = MEM[op[1]]
    elif opcode == 2:
        MEM[op[3]] = MEM[op[1] + MEM[op[2]]]
    elif opcode == 3:
        MEM[op[3]] = MEM[op[1]]
    elif opcode == 4:
        MEM[op[3]] = MEM[op[1]] + MEM[op[2]]
    elif opcode == 5:
        MEM[op[3]] = MEM[op[1]] - MEM[op[2]]
    elif opcode == 6:
        MEM[op[3]] = MEM[op[1]] * MEM[op[2]]
    elif opcode == 7:
        MEM[op[3]] = MEM[op[1]] // MEM[op[2]]
    elif opcode == 8:
        MEM[op[3]] = MEM[op[1]] % MEM[op[2]]
    elif opcode == 9:
        MEM[op[3]] = int(MEM[op[1]] == MEM[op[2]])
    elif opcode == 10:
        MEM[op[3]] = int(MEM[op[1]] > MEM[op[2]])
    elif opcode == 11:
        MEM[op[3]] = int(MEM[op[1]] < MEM[op[2]])
    elif opcode == 12:
        MEM[op[3]] = int(MEM[op[1]] and MEM[op[2]])
    elif opcode == 13:
        MEM[op[3]] = int(MEM[op[1]] or MEM[op[2]])
    elif opcode == 14:
        MEM[op[3]] = int(not MEM[op[1]])
    elif opcode == 20:
        if MEM[op[1]] != 0:
            PC = op[3]
            continue
    elif opcode == 30:
        PC = op[3]
        continue
    elif opcode == 40:
        PC = MEM[op[3]]
        continue
    elif opcode == 50:
        print("output:", MEM[op[1]])
    elif opcode == 60:
        MEM[op[1]] = int(input("input:"))
    else:
        print("illegal opcode:", op)
    PC += 1
    if PC >= len(prog):
        print("end of program reached. halt")
        break

print("total time:", t)
