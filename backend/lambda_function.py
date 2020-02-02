import numpy as np

from qiskit import QuantumCircuit, execute, IBMQ

import json
import boto3


NT = True

JOB = None

def construct_progress(state):
    dm = {}
    dm["action"] = "progress"
    dm["value"] = state
    return json.dumps(dm)

def construct_result(state):
    dm = {}
    dm["action"] = "result"
    dm["value"] = state
    return json.dumps(dm)


# def poll_job_status():
#     while JOB is None:
#         time.sleep(3)
#         print("waiting")
#     while JOB.status != JobStatus.DONE:
#         time.sleep(1)
#         print(JOB.status)

def lambda_handler(event, context):
    apiGW = boto3.client('apigatewaymanagementapi', endpoint_url='https://'+event['requestContext']['domainName'] + '/' + event['requestContext']['stage'], region_name='us-east-1')
    client_id = event['requestContext']['connectionId']
    print("Got client id"  + client_id  )
    client_input = event['body']
    graph = client_input["graph"]
    cost_matrix = np.array(json.loads(graph))
    print(cost_matrix)

    t = 6
    s = 8

    U = calculate_U(cost_matrix)

    IBMQ.load_account()
    
    print("done")
    in_map = simulate_multiple(t,s,U, simulate=False)

    keys = in_map.keys()
    tups = [(key, in_map[key]) for key in keys]
    sorted_arr = sorted(tups, key=lambda tup: int(tup[1], 2))
    winner = sorted_arr[0][0]
    print(sorted_arr)
    print("===")
    print(winner)
    n_list = eigenstr_to_nodes(winner)
    print("Returing " + construct_result(n_list))
    # print(n_list)
    # print(actual_cost(n_list, cost_matrix))
    # print([0,1,2,3], actual_cost([0,1,2,3], cost_matrix))
    # print([1,0,2,3], actual_cost([1,0,2,3], cost_matrix))
    if NT:
        apiGW.post_to_connection(Data=construct_result(n_list), ConnectionId=client_id)
    return {
            'statusCode': 200,
            'body': json.dumps('Hello from Lambda!')
    }



def get_adj():
    N = 4
    cost_matrix = np.zeros((N,N))
    cost_matrix[0,2] = cost_matrix[2,0] = 2
    cost_matrix[3,1] = cost_matrix[1,3] = 3

    test_phases = [[0, np.pi/2 , np.pi /8 , np.pi /4 ], [np.pi /2 , 0, np.pi /4 , np.pi /4 ], [np.pi /8 , np.pi /4 , 0, np.pi /8 ],[ np.pi /4 , np.pi /4 , np.pi /8, 0]]
    test_phases = np.array(test_phases)
    print(test_phases)
    #return cost_matrix
    return test_phases

def calculate_U(cost_matrix):
    B = np.exp(1j *cost_matrix)
    N = len(cost_matrix)

    U = []

    for j in range(N):
        Uj = np.zeros((N,N), dtype=np.complex)
        for k in range(N):
            Uj[k,k] = B[j,k]
        Uj = Uj

        U.append(Uj)

    kron = U[0]
    for i in range(1, len(cost_matrix)):
        kron = np.kron(kron, U[i])

    return U

def uni_gate(phases, x,y,z, name):
    a,b,c,d = phases
    ug = QuantumCircuit(3, name=name)
    ug.cu1((c-a), x,y)
    ug.u1(a, x)
    ug.cu1(b-a, x,z)
    ug.ccx(x,y,z)
    ug.cu1((d-c+a-b)/2, x,z)
    #ug.u1((d-c+a-b)/2, z)
    # ug.ccx(x,y,z)
    ug.ccx(x,y,z)
    ug.cu1((d-c+a-b)/2, x,y)
    ug.cu1((d-c+a-b)/2, x,z)
    return ug.to_instruction()

def qft_dagger(n):
    circ = QuantumCircuit(n, name='I-QFT')
    for j in range(n):
        k = (n-1) - j
        for m in range(k):
            circ.cu1(-np.pi/float(2**(k-m)), k, m)
        circ.h(k)
    return circ.to_instruction()

def construct_circuit(t,s, U, temp):
    
    qc = QuantumCircuit(t+s,t)

    for te in temp:
        qc.x(t+te)

    for t_i in range(t):
        qc.h(t_i)

    test_phases = [0, np.pi/2 , np.pi /8 , np.pi /4 , np.pi /2 , 0, np.pi /4 , np.pi /4 , np.pi /8 , np.pi /4 , 0, np.pi /8 , np.pi /4 , np.pi /4 , np.pi /8, 0]

    for t_i in range(t-1, -1, -1):
        bigU_qc = QuantumCircuit(t+s, name='C-U'.format(2**(t-t_i-1)))
        for idx, u in enumerate(U):
            #phases = (1j *np.log(np.diag(u))).real
            phases = test_phases[idx*4 : idx*4 + 4]
            # print(idx, (t-t_i-1)*4)
            gate = uni_gate(phases, 0,1,2, name='uni-{}'.format(idx))
            #for power in range(2**(t-t_i-1)):
            #    print(power)
            bigU_qc.append(gate, [t_i,idx*2+t,idx*2+t+1])
        for power in range(2**(t-t_i-1)):
            qc.append(bigU_qc, range(t+s))
        qc.barrier()
            
        #qc.append(bigU_qc.to_instruction(), range(t+s))

    # qft_dagger(qc, t)
    qc.append(qft_dagger(t), range(t))

    for t_i in range(t):
        qc.measure(t_i, t_i)
    return qc

def run_qc(t,s,U, eigenstate, simulate=True):
    qc = construct_circuit(t,s,U,eigenstate)
    if simulate:
        backend = Aer.get_backend('qasm_simulator')
    else:
        print("Using ibmq")
        provider = IBMQ.get_provider(group='open')
        backend = provider.get_backend('ibmq_qasm_simulator')

    JOB = execute(qc, backend, shots=1024)
    result = JOB.result()

    from qiskit.visualization import plot_histogram

    counts = result.get_counts()

    max_key = max(counts, key=counts.get)

    return max_key

def simulate_multiple(t,s,U,simulate=True):
    eigenstates = [[0, 1, 5, 6], [1, 2, 4, 5], [0, 4, 5, 7], [1, 2, 3, 6], [0, 1, 2, 7], [0, 2, 3, 5]]

    res = {}
    for e in eigenstates:
        e_i = ["0" for _ in range(8)]
        for i in e:
            e_i[i] = "1"
        res["".join(e_i)] = run_qc(t,s,U,e, simulate=simulate)

    
    return res

def eigenstr_to_nodes(e):
    nodes = []
    for i in range(0,8,2):
        b = e[i] + e[i+1]
        b = int(b,2)
        nodes.append(b)

    return nodes

def actual_cost(nodes_list, U):
    s = 0

    log = []
    for idx,node in enumerate(nodes_list):
        n_node = nodes_list[(idx+1) % len(nodes_list)]
        s = s + U[node][n_node]
        log.append(U[node][n_node])

    print(log)

    return s
    



if __name__ == '__main__':
    t = 6
    s = 8

    cost_matrix = get_adj()
    U = calculate_U(cost_matrix)

    in_map = simulate_multiple(t,s,U)

    keys = in_map.keys()
    tups = [(key, in_map[key]) for key in keys]
    sorted_arr = sorted(tups, key=lambda tup: int(tup[1], 2))
    winner = sorted_arr[0][0]
    print(sorted_arr)
    print("===")
    print(winner)
    n_list = eigenstr_to_nodes(winner)
    print(n_list)
    print(actual_cost(n_list, cost_matrix))
    print([0,1,2,3], actual_cost([0,1,2,3], cost_matrix))
    print([1,0,2,3], actual_cost([1,0,2,3], cost_matrix))

