import numpy as np
from scipy.stats import uniform
import copy
from itertools import product
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt
import itertools
from matplotlib.patches import Patch # Needed for plotting
import winsound
import random

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath}",
        "font.family": "Times New Roman",
    }
)

light_grey = '#D3D3D3'

TEST_RUN = False # Set to true for code execution.

random.seed(8229377)

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
program_start_time = time.time()

f = open("00_program_output.txt", "w")

def yield_states(state_count):
    # state_count is a tuple with at the i-th location the number of states in
    # position i.
    
    states = len(state_count)
     
    no_states = np.array(state_count)
    
    counter = [0]*states
    
    while not counter[0]==state_count[0]:
        
        yield tuple(counter)
        
        counter[-1]+=1
        
        for i in reversed(range(1,states)):
            if counter[i]==no_states[i]:
                counter[i-1]+=1
                counter[i]=0
        
    return None

def product_transitions(Q_array_list):
    # List of matrices as np.arrays
    
    Q = dict()
    
    states_array = [len(Q_matrix) for Q_matrix in Q_array_list]
        
    for y1 in yield_states(states_array):
        Q[y1] = dict()
        
        for y2 in yield_states(states_array):
            # Q[y1][y2] = 
            prob = 1
            for i, P in enumerate(Q_array_list):
                prob *= P[y1[i]][y2[i]]
            
            Q[y1][y2] = prob
    
    return Q

def dominate(vct, vectors, index=-1, d_min=0):
    # vct represent a vector in vectors.
    # vectors is a list of numpy arrays that represent vectors.
    # index is the index position of vct in vectors (if it is in there).
    # min_d is a measure of dominance.
    
    # This function follows Figure 2 in Cassandra et al. (1997).

    m = gp.Model("Dominate")
    
    # Add variables.
    d = m.addVar(lb=0, name="d")
    
    x = m.addMVar(len(vct), lb=0) # belief state

    m.setObjective(d, GRB.MAXIMIZE)
    
    for i, v in enumerate(vectors):
        if i == index:
            continue
        
        m.addConstr(x @ vct<=d+x @ v)
    
    ones = np.array([1]*len(vct))
        
    m.addConstr(x @ ones==1)
    
    m.Params.OutputFlag = 0
    m.optimize()
    
    # Info: https://www.gurobi.com/documentation/10.0/examples/lp_py.html
    if m.Status == GRB.INF_OR_UNBD:
        m.setParam(GRB.Param.Presolve, 0)
        m.optimize()

    if m.Status == GRB.INFEASIBLE:
        return False, None

    if m.ObjVal>d_min:
        solution = m.getAttr('x', m.getVars())
        return True, np.array(solution[1:])

    return False, None

def argmin(x,F):
    
    mn = float('inf')
    mn_index = 0
    
    for i, v in enumerate(F):
        
        if np.matmul(x, v) < mn:
            mn = np.matmul(x, v)
            mn_index = i
            
    return mn_index

def filter_vectors(F, d_min = 0):
    
    vct_length = len(F[0])
    
    W = []
    
    for i in range(vct_length):
        vct = [0]*vct_length
        vct[i] = 1
        pure_vct = np.array(vct)
        
        mn_index = argmin(pure_vct,F)
        mn_vct = F[mn_index]
        W.append(mn_vct)
        del F[mn_index]
        
        # No vectors left to cut out.
        if len(F)==0:
            return W
        
    while len(F)>0:
        vct = F[0]

        parallel, x = dominate(vct, W, 0, d_min)
        
        if parallel == False:
            del F[0]
        else:
            mn_index = argmin(x,F)
            mn_vct = F[mn_index]
                        
            W.append(mn_vct)
            del F[mn_index]
                      
    return W

def tau(C,Q,Y,discount,alpha,y):
    
    observations = len(Y[0])
    states = len(Q)
    
    term_1 = (1/observations)*C    
    term_2 = discount*sum([alpha[nxt_s]*Y[nxt_s,y] * Q[:,nxt_s] for nxt_s in range(states)])
    
    return term_1 + term_2

def FindBeliefDec(vct,vectors):
    c = uniform.rvs(size=len(vct))
    c = np.array(c)
    c = c/sum(c)

    if len(vectors)==0:
        return c, float('inf')

    m = gp.Model("Dominate")
    
    # Add variables.
    d = m.addVar(lb=0, name="d")
    x = m.addMVar(len(vct), lb=0) # belief state
    ones = np.array([1]*len(vct))
    m.addConstr(x @ ones==1)

    m.setObjective(d, GRB.MAXIMIZE)
    
    W = []
    b = c
    first_loop = True
    
    while (not np.array_equal(b,c)) or first_loop:
        first_loop = False
        
        b = c
        
        min_val = float('inf')
        v_min = []
        
        for v in vectors:
            val = v @ b - vct @ b 
            if val < min_val:
                min_val = val
                v_min = v
        
        if len(v_min)>0:
            W.append(v_min)
        else:
            print("Unforeseen case")
        
        m.addConstr(vct @ x + d <= v_min @ x)
                
        m.Params.OutputFlag = 0
        m.optimize()
        
        solution = m.getAttr('x', m.getVars())
        c = np.array(solution[1:])
        d_opt = solution[0]

    if d_opt > 0:
        return c, d_opt
    else:
        return [], d_opt

def best_vector(b,vectors):
    
    mn_val = float('inf')
    v_mn = []
    i_mn = -1
    for i, v in enumerate(vectors):
        if mn_val > b @ v:
            mn_val = b @ v
            v_mn = v
            i_mn = i

    return v_mn, i_mn

def pruning_Walraven(vectors, d_min = 0.001):
        
    pruned_W = []
    W = copy.deepcopy(vectors)
    
    states = len(vectors[0])
    
    while len(W)>0:
        
        w = W[-1]
        
        is_dominated = False
        
        for v in pruned_W:
        
            dominates_w = True # Set to false if w is larger at a corner.
        
            for s in range(states):
                if v[s]>=w[s]:
                    dominates_w = False
                    
            if dominates_w == True:
                W.pop() # remove w.
                is_dominated = True
                break
            
        if is_dominated:
            continue
                
        b, d_opt = FindBeliefDec(w,pruned_W)
        
        if len(b)==0 or d_opt<d_min:
            W.pop()
        else:
            w, index_w = best_vector(b, W)
            W.pop(index_w)
            pruned_W.append(w)
                    
    
    return pruned_W

def direct_sum(dct):
    #Input is a dict with lists.
    
    itr_list = []
    drct_sums = []
    
    
    for k in dct.keys():
        itr_list.append(len(dct[k]))

    for y in yield_states(itr_list):
        
        sm = 0
        
        for i in range(len(itr_list)):
            sm += dct[i][y[i]]
            
        drct_sums.append(sm)

    return drct_sums


def min_value(alphas):
    # Find the minimum value corresponding the the piecewise linear function
    # of the maximum of all alpha vectors.
    
    m = gp.Model("Minimize")
    
    # Add variables.
    d = m.addVar(name="d")
    
    x = m.addMVar(len(alphas[0]), lb=0)

    m.setObjective(d, GRB.MINIMIZE)
    
    for a in alphas:
        m.addConstr(d >= x @ a)
    
    ones = np.array([1]*len(alphas[0]))
        
    m.addConstr(x @ ones==1)
    
    m.Params.OutputFlag = 0
    m.optimize()
    
    solution = m.getAttr('x', m.getVars())
    return solution[0] # Minimum value of value function.
    


def vector_pruning_Walraven(R,P,O,y,alphas, accuracy_target=0.1, dmin = 0.01, max_run_time = 3600):

    start_time = time.time()    

    max_c = float('-inf')
    for i, r in enumerate(R):
        if max_c < abs(r.max()):
            max_c = abs(r.max())

    observations = len(O[0][0]) # total observation states.
    Saz = {} # Will contain the vectors per action.
    bound = float('inf')
    
    n=0
    while bound>accuracy_target: 
        n+=1
        
        Sz = {}
        
        for a in range(len(P)):
                    
            for z in range(observations):
                
                list_az = []
                
                for alpha in alphas:
                    
                    list_az.append(tau(R,P,O,y,alpha,a,z))
                    
                Sz[z] = pruning_Walraven(list_az,d_min=dmin)
                
            Saz[a] = pruning_Walraven(direct_sum(Sz),d_min=dmin)
    
        S = []
        for a in Saz.keys():
            S += Saz[a] 
    
        alphas = pruning_Walraven(S,d_min=dmin)
        bound = max_c*(y**(n+1)/(1-y)) # Theorem 7.6.3, Krishnamurthy
        
        print("\n>. Iteration {}, vectors {}, bound {}".format(n,len(alphas),
                                                          bound))
        running_time = time.time() - start_time
        print("Running time {}\n".format(running_time))
        
        if running_time>max_run_time:
            break
        
    print("Minimal value {}\n".format(min_value(alphas)))
    
    running_time = time.time() - start_time

    return alphas, Saz, running_time

def vector_pruning_modified(Q,Y,C,C_insp,C_replace,discount_factor,
                            accuracy_target=0.1,dmin=0.001,max_run_time=1000,
                            analyze_time = False):

    # Number of degradation states per component.    
    state_number = [0]*len(next(iter(Q.keys())))

    for k in Q.keys():
        for i, val in enumerate(k):
            state_number[i] = max(state_number[i],val+1)

    # Create dictionaries relating component based states to linear numbered states.
    state_to_tuple = dict()
    tuple_to_state = dict()
    
    for i, v in enumerate(yield_states(state_number)):
        state_to_tuple[i] = v
        tuple_to_state[v] = i 
    
    # Calculate the number of states
    total_states = 1
    for i, _ in enumerate(state_number):
        total_states*=state_number[i]
    
    # Calculate the number of observations
    observations = len(Y[tuple([0]*len(state_number))])

    # Give a name to the problem
    dgr_state_strings = [str(s) for s in state_number]
    name = "obsv_"+str(observations)+"_dgr_"+"_".join(dgr_state_strings)
    timing_dict = {"No intervention time":[],"Intervention time":[]}

    
    # Change the matrices to the linear numbering of states.
    C_flat = np.zeros(total_states)
    for y in yield_states(state_number):
        s = tuple_to_state[y]
        C_flat[s] = C[y]
    
    Y_flat = np.zeros((total_states,observations))

    for y in yield_states(state_number):
        s = tuple_to_state[y]
        for o in range(observations):
            Y_flat[s,o] = Y[y][o]
    
    Q_flat = np.zeros((total_states,total_states))
    
    for y1 in yield_states(state_number):
        s1 = tuple_to_state[y1]
        for y2 in yield_states(state_number):
            s2 = tuple_to_state[y2]
            Q_flat[s1,s2] = Q[y1][y2]

    # Definitions wrapped up. Move to the algorithm.
    start_time = time.time()

    max_c = float('-inf')
    for i, c in enumerate(C_flat):
        if max_c < abs(c.max()):
            max_c = abs(c.max())+discount_factor*(C_insp+sum(C_replace))
            # max_c = abs(c.max())
    
    n=0
    bound = float('inf')
    alphas = [np.array([0]*total_states)]

    while bound>accuracy_target:
        # break
        n+=1
        
        # Calculate the value of the maintenance intervention action.
        start_time_intervention = time.time()
        
        opt_val = dict()
        opt_val_flat = np.array([0]*total_states)
        
        for y in yield_states(state_number):
            
            action_values = []
            
            for r in product([0,1], repeat = len(state_number)):
                # Calculate variable cost
                C_v = sum([r[i]*C_replace[i] for i in range(len(state_number))])
                                
                # Determine the next state when apply action r to core state y.
                y_next = tuple([(1-r[i])*y[i] for i in range(len(y))])
                s = tuple_to_state[y_next]
                
                new_belief = np.array([0]*total_states)
                new_belief[s] = 1
                
                long_term_cost = float('inf')
                
                for a in alphas:
                    if long_term_cost>new_belief @ a:
                        long_term_cost=new_belief @ a
                        
                action_val = C_v + long_term_cost
                        
                action_val = discount_factor*action_val
    
                action_values.append(action_val)
    
            opt_val[y] = min(action_values)
            s = tuple_to_state[y]
            opt_val_flat[s] = min(action_values)
         
        inspection_alpha = copy.deepcopy(C_flat)
        inspection_alpha += np.array([discount_factor*C_insp]*total_states)
        inspection_alpha += Q_flat @ opt_val_flat
        
        timing_dict['Intervention time'].append(time.time()-start_time_intervention)
        
        # Calcualte the value of the no intervention action.
        start_time_no_intervention = time.time()


        Sz = {}
        
        for y in range(observations):
            
            list_az = []
            
            for alpha in alphas:
                
                list_az.append(tau(C_flat,Q_flat,Y_flat,discount_factor,alpha,y))
                
            Sz[y] = pruning_Walraven(list_az,d_min=dmin)
                
        S = direct_sum(Sz)
    
        alphas = pruning_Walraven(S,d_min=dmin)
        
        timing_dict['No intervention time'].append(time.time()-start_time_no_intervention)

        alphas = alphas + [inspection_alpha]
       
        bound = max_c*(discount_factor**(n+1)/(1-discount_factor)) # Theorem 7.6.3, Krishnamurthy
        
        print("\n>. Iteration {}, vectors {}, bound {}".format(n,len(alphas),
                                                          bound))
        running_time = time.time() - start_time
        print("Running time {}\n".format(running_time))
        
        if bound <= accuracy_target:
            f.write("For problem with state number: {}\n".format(state_number))
            f.write(">. Iteration {}, vectors {}, bound {}\n".format(n,len(alphas),bound))
        
        
        if running_time>max_run_time:
            f.write("For problem with state number: {}\n".format(state_number))
            f.write(">. Iteration {}, vectors {}, bound {}\n".format(n,len(alphas),bound))

            break
        
    print("Minimal value {}".format(min_value(alphas)))
    f.write("Minimal value {}\n".format(min_value(alphas)))
    
    starting_belief_state = [0.0 for i in alphas[0]]
    starting_belief_state[0]=1.0
    
    
    starting_value = float('inf')
    for a in alphas:
        if starting_value > a @ starting_belief_state:
            starting_value = a @ starting_belief_state
    
    f.write("Starting value {} in state 0\n".format(starting_value))
    
    running_time = time.time() - start_time
    print("Running time {}".format(running_time))
    f.write("Total running time {}\n".format(running_time))
    f.write("Average time in NO intervention {}\n".format(np.average(timing_dict['No intervention time'])))
    f.write("Average time in intervention {}\n".format(np.average(timing_dict['Intervention time'])))

    return alphas, inspection_alpha, state_to_tuple

def basic_checks():
    
    # Simply check of FindBeliefDec.
    vct = np.array([0,0,0])
    vectors = [np.array([1,1,2]),np.array([1,2,1])]
    vct = np.array([2,2,2])
    vectors = [np.array([1,1,2]),np.array([1,2,1])]
    print(FindBeliefDec(vct,vectors))
    vct = np.array([5,0,1])
    vectors = [np.array([0,0,0])]
    print(FindBeliefDec(vct,vectors))
    
    # Check best_vector.
    blf = np.array([0.5,0.5,1])
    vectors = [np.array([1,1,2]),np.array([1,2,1])]
    print(best_vector(blf,vectors))
    
    # Check pruning_Walraven.
    vectors = [np.array([1,1,2]),np.array([1,2,1]),np.array([2,2,2]),np.array([5,0,1]),np.array([0,0,0])]
    print(pruning_Walraven(vectors))

    return 0


def rand_from_array(arr):
    pass

class simulator:
    def __init__(self,Q,Y,C,C_insp,C_replace,discount_factor,alphas=[],
                 inpsection_alpha=np.array([])):

        print("Creating simulator")
        
        self.Q = Q
        self.Y = Y
        self.C = C 
        self.C_insp = C_insp
        self.C_replace = C_replace
        self.discount_factor = discount_factor
        
        # Number of degradation states per component.
        self.state_number = [0]*len(next(iter(Q.keys())))
    
        for k in Q.keys():
            for i, val in enumerate(k):
                self.state_number[i] = max(self.state_number[i],val+1)
    
        # Create dictionaries relating component based states to linear numbered states.
        self.state_to_tuple = dict()
        self.tuple_to_state = dict()
        
        for i, v in enumerate(yield_states(self.state_number)):
            self.state_to_tuple[i] = v
            self.tuple_to_state[v] = i 
                
        for k in Q.keys():
            for i, val in enumerate(k):
                self.state_number[i] = max(self.state_number[i],val+1)

        total_states = len(Q.keys())
        self.total_states = len(Q.keys())
        observations = len(Y[tuple([0]*len(self.state_number))])

        # If no value vectors are specified create them.
        if alphas == []:
            dummy_alpha_1 = np.array([0]*total_states)
            dummy_alpha_1[0] = 1
            alphas = [dummy_alpha_1]
        
        if inpsection_alpha.size == 0:
            dummy_alpha_2 = np.array([0]*total_states)
            dummy_alpha_2[total_states-1] = 1
            inspection_alpha = dummy_alpha_2
            
        self.alphas = alphas
        self.inspection_alpha = inspection_alpha
            
        self.Y_flat = np.zeros((total_states,observations))

        for y in yield_states(self.state_number):
            s = self.tuple_to_state[y]
            for o in range(observations):
                self.Y_flat[s,o] = Y[y][o]
        
        self.Q_flat = np.zeros((total_states,total_states))
        
        for y1 in yield_states(self.state_number):
            s1 = self.tuple_to_state[y1]
            for y2 in yield_states(self.state_number):
                s2 = self.tuple_to_state[y2]
                self.Q_flat[s1,s2] = Q[y1][y2]
        
        return None
        
    def hitting_analysis(self):
        
        f.write("\nTesting belief state updating\n")
        
        observations = len(self.Y[tuple([0]*len(self.state_number))])

        action_dict = {0:"no intervention", 1:"maintenance intervention"}

        for observation in range(observations):
        
            pi = np.array([0]*self.total_states)
            pi[0] = 1
            
            switch_timing = []
            switch_actions = []
            switch_beliefs = []
            
            optimal_action = 0
            
            for i in range(100):
                pi = self.belief_state_update(pi,observation)
                
                if optimal_action != self.determine_optimal_action(pi):
                    optimal_action = self.determine_optimal_action(pi)
                    
                    switch_timing.append(i+1)
                    switch_actions.append(optimal_action)
                    switch_beliefs.append(pi)

            f.write("\nObservation {}\n".format(observation))
            for i, beliefs in enumerate(switch_beliefs):
                f.write("Switch to action {} after {} obsv\n".format(switch_actions[i],\
                                                                     switch_timing[i]))
                for j, prob in enumerate(beliefs):
                    
                    
                    f.write("{}: {:.03f} ".format(self.state_to_tuple[j],prob))
                f.write("\n")

    def determine_optimal_action(self,pi):
        
        # If continue is optimal set the optimal action to continue.
        optimal_action = 1
        
        for a in self.alphas:
            if a @ pi < self.inspection_alpha @ pi:
                optimal_action = 0
        
        return optimal_action
        
    def set_replacement_policy(self):
        pass
    
    def belief_state_update(self, pi, observation):
        
        new_pi = np.array([0.0]*self.total_states)
        
        norm_factor = (pi @ self.Q_flat @ self.Y_flat)[observation]
        
        for t in range(self.total_states):
            sum_t = (pi @ self.Q_flat)[t]*self.Y_flat[t][observation]/norm_factor
            new_pi[t] = sum_t
        
        return new_pi
    
    def simulation_run(self,start_pi,steps):
        pass


def num_exp_1_in_paper():
    
    # This is an example built from 2 TP2 degradation matrices and a with an
    # ObsvTP2 matrix.
    # 2x2 degradation states and 3 observation states.
    # Remark that this is slower to solve than the 3x3 dgr. state, 2 obsv example.
    # The observation states slow things down.
    
    # Define transition matrices from tuple state to tuple state.
    C = dict() # Operating cost
    C[(0,0)] = 0
    C[(0,1)] = 1 
    C[(1,0)] = 2 
    C[(1,1)] = 15.5
    
    C_insp = 10
    C_replace = dict()
    C_replace[0] = 2
    C_replace[1] = 3
    
    discount_factor = 0.9
    
    # 1st set of matrices.
    
    Y = dict() # Observation matrix
    Y[(0,0)] = dict()
    Y[(0,1)] = dict()
    Y[(1,0)] = dict()
    Y[(1,1)] = dict()

    Y[(0,0)][0] = 0.8
    Y[(0,0)][1] = 0.2
    Y[(0,0)][2] = 0

    Y[(0,1)][0] = 0.2
    Y[(0,1)][1] = 0.6
    Y[(0,1)][2] = 0.2
    
    Y[(1,0)][0] = 0.2
    Y[(1,0)][1] = 0.8
    Y[(1,0)][2] = 0.0
    
    Y[(1,1)][0] = 0
    Y[(1,1)][1] = 0.2
    Y[(1,1)][2] = 0.8
    
    # Define the degradation matrix as transitions from states to states.
    Q = dict()
    Q[(0,0)] = dict()
    Q[(0,1)] = dict()
    Q[(1,0)] = dict()
    Q[(1,1)] = dict()
    
    Q[(0,0)][(0,0)] = 0.9*0.8
    Q[(0,0)][(0,1)] = 0.9*0.2
    Q[(0,0)][(1,0)] = 0.1*0.8
    Q[(0,0)][(1,1)] = 0.1*0.2

    Q[(0,0)][(0,0)] = 0.9*0.8
    Q[(0,0)][(0,1)] = 0.9*0.2
    Q[(0,0)][(1,0)] = 0.1*0.8
    Q[(0,0)][(1,1)] = 0.1*0.2
    
    Q[(0,1)][(0,0)] = 0
    Q[(0,1)][(0,1)] = 0.9
    Q[(0,1)][(1,0)] = 0
    Q[(0,1)][(1,1)] = 0.1
    
    Q[(1,0)][(0,0)] = 0
    Q[(1,0)][(0,1)] = 0
    Q[(1,0)][(1,0)] = 0.8
    Q[(1,0)][(1,1)] = 0.2
    
    Q[(1,1)][(0,0)] = 0
    Q[(1,1)][(0,1)] = 0
    Q[(1,1)][(1,0)] = 0
    Q[(1,1)][(1,1)] = 1
    
    resolution = 1000
    
    sim = simulator(Q,Y,C,C_insp,C_replace,discount_factor)
    
    if TEST_RUN: # Set to true for a fast run
        discount_factor = 0.5
        resolution = 50
    
    accuracy_target = 0.2
    dmin = 0.05
    
    alphas, inspection_alpha, state_to_tuple = vector_pruning_modified(Q, Y, C, C_insp, C_replace, discount_factor,
                                                                       accuracy_target=accuracy_target,dmin=dmin)

    # Make a plot showing the optimal policy.
    
    figs, axs = plt.subplots(4,3,figsize=(6,6),dpi=300,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 1],
                                              width_ratios=[1, 1, 1])) 
    plt.setp(axs, xticks=[0,1], yticks=[0,1])
    
    figs.tight_layout(h_pad=2.2, w_pad=-5)
    
    # Make a legend
    legend_elements = [Patch(facecolor='steelblue', label='No intervention'),
                        Patch(facecolor='orange', label='Intervention'),
                        Patch(facecolor=light_grey, label='Infeasible')]
    axs[0][0].legend(handles=legend_elements, bbox_to_anchor=(0.13,0.94), 
              loc='upper left', fontsize=10)
    axs[0,0].set_axis_off()


    for i_00, belief_00 in enumerate(np.linspace(0,1,11)[::-1]):
        # Draw trangles in the bottom right corner (1,1) is the bottom right point.
        
        # Specify the grid on which to draw the data.
        X = np.linspace(0,1,resolution+1)
        Y = np.linspace(0,1,resolution+1)
        X_Mesh, Y_Mesh = np.meshgrid(X, Y)
        Z = 0*(X_Mesh+Y_Mesh)
        
        for x, y in itertools.product(enumerate(X),enumerate(Y)):
            belief_01 = x[1]
            belief_10 = y[1]
            i_01 = x[0]
            i_10 = y[0]
            
            belief_11 = 1-belief_00-belief_01-belief_10
            if belief_11<0:
                Z[i_01][i_10] = 2
                continue
            
            belief_state = np.array([belief_00,belief_01,
                                     belief_10,belief_11])
                    
            optimal_action = 0 # inspect
            
            for a in alphas:
                if optimal_action == 1:
                    break
                
                # If continue is optimal set the optimal action to continue.
                if a @ belief_state < inspection_alpha @ belief_state:
                    optimal_action = 1

            Z[i_01][i_10] = optimal_action
            
        
        
        colors = ['orange','steelblue',light_grey]
        levels = [-0.5,0.5,1.5,2.5]
                
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
        # colors below specifies the colors the different regions are.
        
        pos_x = (i_00+1)//3
        pos_y = (i_00+1)%3
        
        axs[pos_x][pos_y].set_title(r"$\pi_{(0,0)}$" + " = {:.02f}".format(belief_00),
                                    size = 10)
        axs[pos_x][pos_y].set_xlabel(r"$\pi_{(0,1)}$", size = 9)
        axs[pos_x][pos_y].set_ylabel(r"$\pi_{(1,0)}$", size = 9)

        axs[pos_x][pos_y].xaxis.set_label_coords(.5, -.2)
        axs[pos_x][pos_y].yaxis.set_label_coords(-.2, .5)

        axs[pos_x][pos_y].set_aspect('equal', 'box')

        CS = axs[pos_x][pos_y].contourf(X_Mesh, Y_Mesh, Z, colors = colors, levels = levels)
            
    plt.savefig('00_Exp1_Stage1_plot.png', bbox_inches="tight")
    plt.show()
    # Detemine optimal stage 2 actions
    
    policy = dict()
    
    # Number of degradation states per component.
    state_number = [0]*len(next(iter(Q.keys())))

    for k in Q.keys():
        for i, val in enumerate(k):
            state_number[i] = max(state_number[i],val+1)

    # Create dictionaries relating component based states to linear numbered states.
    state_to_tuple = dict()
    tuple_to_state = dict()
    
    for i, v in enumerate(yield_states(state_number)):
        state_to_tuple[i] = v
        tuple_to_state[v] = i 

    # Calculate the number of states
    total_states = 1
    for i, _ in enumerate(state_number):
        total_states*=state_number[i]

    for y in yield_states(state_number):
        
        action_values = []
        optimal_action = [0]*len(state_number)
        optimal_value = float('inf')
        
        for r in product([0,1], repeat = len(state_number)):
            # Calculate variable cost
            
            C_v = sum([r[i]*C_replace[i] for i in range(len(state_number))])
            
            # Determine the next state when apply action r to core state y.
            y_next = tuple([(1-r[i])*y[i] for i in range(len(y))])
            s = tuple_to_state[y_next]
            
            new_belief = np.array([0]*total_states)
            new_belief[s] = 1
            
            long_term_cost = float('inf')
            
            for a in alphas:
                if long_term_cost>new_belief @ a:
                    long_term_cost=new_belief @ a
                    
            action_val = C_v + long_term_cost
                    
            action_val = discount_factor*action_val

            if action_val < optimal_value:
                optimal_value = action_val
                optimal_action = r

        policy[y]=optimal_action
        
        
    state_action_target = []
    for key in policy.keys():
        target = []
        for i, k in enumerate(key):
            target.append(k*(1-policy[key][i]))
        
        state_action_target.append([key,policy[key],target])
        
    for L in state_action_target:
        print("{} -{}-> {}".format(L[0],L[1],L[2]))
        
    f.write("\nStage II: state, action, target\n")
    # Format the stage 2 optimal policy for LaTeX
    for L in state_action_target:
        f.write("{} & {} & {}\\\\\n".format(L[0],L[1],tuple(L[2])))
    
    
    # Determine starting value
    starting_belief = np.array([0.0 for i in range(len(alphas[0]))])
    starting_state = tuple_to_state[(0,0)]
    starting_belief[starting_state] = 1.0
    
    starting_value = float('inf')
    for a in alphas:
        if starting_value > starting_belief @ a:
            starting_value = starting_belief @ a
    
    f.write("Outside check: starting in good state value {}\n".format(starting_value))

    
    ### Perfect information case ###
    Y = dict()
    
    for x in product(range(2),repeat=2):
        Y[x] = dict()
    
    # Set observation matrix to perfect information.
    for x in product(range(2),repeat=2):
        for y in product(range(2),repeat=2):
            if x == y:
                Y[x][tuple_to_state[y]] = 1
            else:
                Y[x][tuple_to_state[y]] = 0

    f.write("\nPerfect information case\n")
    alphas, inspection_alpha, state_to_tuple = vector_pruning_modified(Q, Y, C, C_insp, C_replace, discount_factor, 
                                                                       accuracy_target=0.1,dmin=0.001)


    # Determine starting value
    starting_belief = np.array([0.0 for i in range(len(alphas[0]))])
    starting_state = tuple_to_state[(0,0)]
    starting_belief[starting_state] = 1.0
    
    starting_value = float('inf')
    for a in alphas:
        if starting_value > starting_belief @ a:
            starting_value = starting_belief @ a
    
    f.write("Outside check: starting in good state value {}\n".format(starting_value))


    ### No information case ###
    for x in product(range(2),repeat=2):
        Y[x] = dict()
        Y[x][0] = 1

    f.write("\nNo information case\n")
    alphas, inspection_alpha, state_to_tuple = vector_pruning_modified(Q, Y, C, C_insp, C_replace, discount_factor,
                                                                       accuracy_target=0.1,dmin=0.001)
    
    
    # Determine starting value
    starting_belief = np.array([0.0 for i in range(len(alphas[0]))])
    starting_state = tuple_to_state[(0,0)]
    starting_belief[starting_state] = 1.0
    
    starting_value = float('inf')
    for a in alphas:
        if starting_value > starting_belief @ a:
            starting_value = starting_belief @ a
    
    f.write("Outside check: starting in good state value {}\n".format(starting_value))

    sim.inspection_alpha = inspection_alpha
    sim.alphas = alphas
    sim.hitting_analysis()
    
    return 0

def num_exp_3_w_plots_in_paper():
    
    # Add a comparison to complete and no information.
    
    # This is an example built from 2 TP2 degradation matrices and a with an
    # but without an ObsvTP2 matrix.
    # 3x3 degradation states and 3 observation states.
    # Remark: this seems to go fine. Scale up to 3x3x3 and 3 obsv.
    
    # Define transition matrices from tuple state to tuple state.
    C = dict() # Operating cost
    C[(0,0)] = 0
    C[(0,1)] = 2 # 0.25
    C[(0,2)] = 10
    C[(1,0)] = 2 # 0.25
    C[(2,0)] = 10
    C[(1,1)] = 2 # 0.25
    C[(1,2)] = 10
    C[(2,1)] = 10
    C[(2,2)] = 10
    
    C_insp = 10
    C_replace = dict()
    C_replace[0] = 5
    C_replace[1] = 500 # 50
    
    discount_factor = 0.9
    
    # Initialize degradation matrix and set all terms to 0.
    Q = dict()
    for x in product(range(3),repeat=2):
        Q[x]=dict()
        for y in product(range(3),repeat=2):
            Q[x][y]=0
        
    # Transition with correlation. If one component is degraded further
    # then the other component degrades faster (+0.1).
    
    rate1 = 0.2 # Degradation rate of component 1
    rate2 = 0.02 # Degradation rate of component 2
    
    Q1 = [[1-rate1,rate1,0],[0,1-rate1,rate1],[0,0,1]]
    Q2 = [[1-rate2,rate2,0],[0,1-rate2,rate2],[0,0,1]]
    
    Q[(0,0)][(0,0)] = Q1[0][0]*Q2[0][0]
    Q[(0,0)][(1,0)] = Q1[0][1]*Q2[0][0]
    Q[(0,0)][(0,1)] = Q1[0][0]*Q2[0][1]
    Q[(0,0)][(1,1)] = Q1[0][1]*Q2[0][1]
    
    Q[(1,0)][(1,0)] = Q1[1][1]*Q2[0][0]
    Q[(1,0)][(2,0)] = Q1[1][2]*Q2[0][0]
    Q[(1,0)][(1,1)] = Q1[1][1]*Q2[0][1]
    Q[(1,0)][(2,1)] = Q1[1][2]*Q2[0][1]
    
    Q[(2,0)][(2,0)] = Q1[2][2]*Q2[0][0]
    Q[(2,0)][(2,1)] = Q1[2][2]*Q2[0][1]
    
    Q[(0,1)][(0,1)] = Q1[0][0]*Q2[1][1]
    Q[(0,1)][(1,1)] = Q1[0][1]*Q2[1][1]
    Q[(0,1)][(0,2)] = Q1[0][0]*Q2[1][2]
    Q[(0,1)][(1,2)] = Q1[0][1]*Q2[1][2]
    
    Q[(1,1)][(1,1)] = Q1[1][1]*Q2[1][1]
    Q[(1,1)][(2,1)] = Q1[1][2]*Q2[1][1]
    Q[(1,1)][(1,2)] = Q1[1][1]*Q2[1][2]
    Q[(1,1)][(2,2)] = Q1[1][2]*Q2[1][2]
        
    Q[(2,1)][(2,1)] = Q1[2][2]*Q2[2][1]
    Q[(2,1)][(2,2)] = Q1[2][2]*Q2[2][2]
    
    Q[(0,2)][(0,2)] = Q1[0][0]*Q2[2][2]
    Q[(0,2)][(1,2)] = Q1[0][1]*Q2[2][2]
    
    Q[(1,2)][(1,2)] = Q1[1][1]*Q2[2][2]
    Q[(1,2)][(2,2)] = Q1[1][2]*Q2[2][2]
        
    Q[(2,2)][(2,2)] = 1
    
    Y = dict()
    
    for i in range(3):
        for j in range(3):
            Y[(i,j)] = dict()
    
    Y[(0,0)][0] = 1
    Y[(0,0)][1] = 0
    
    Y[(0,1)][0] = 3/4
    Y[(0,1)][1] = 1/4
    
    Y[(0,2)][0] = 1/2
    Y[(0,2)][1] = 1/2
    
    Y[(1,0)][0] = 3/4
    Y[(1,0)][1] = 1/4
    
    Y[(1,1)][0] = 2/4
    Y[(1,1)][1] = 2/4
    
    Y[(1,2)][0] = 1/4
    Y[(1,2)][1] = 3/4
    
    Y[(2,0)][0] = 1/2
    Y[(2,0)][1] = 1/2
    
    Y[(2,1)][0] = 1/4
    Y[(2,1)][1] = 3/4
    
    Y[(2,2)][0] = 0
    Y[(2,2)][1] = 1

    sim = simulator(Q,Y,C,C_insp,C_replace,discount_factor)

    ### GENERATE PLOTS ###

    resolution = 1000
    accuracy_target = 0.2
    dmin=0.05
    
    if TEST_RUN:
        f.write("TEST RUN AT EXP 3")
        # resolution = 50
        resolution = 50
        accuracy_target=10
        dmin = 3

    f.write("Imperfect information case\n")
    alphas, inspection_alpha, state_to_tuple = vector_pruning_modified(Q, Y, C, C_insp, C_replace, discount_factor, accuracy_target=accuracy_target,
                                                                       dmin=dmin,analyze_time=True)
              
    number_of_states = len(alphas[0])
            
    corner_points = [[(0,0),(2,0),(2,2)]] # Definitely gives a 3-region pol.                             
    # corner_points = [[(0,0),(1,0),(2,2)]]                                       
    # First point is bottom left corner, second bottom right, third, top left.
    
    # Number of degradation states per component.
    state_number = [0]*len(next(iter(Q.keys())))

    for k in Q.keys():
        for i, val in enumerate(k):
            state_number[i] = max(state_number[i],val+1)

    # Create dictionaries relating component based states to linear numbered states.
    state_to_tuple = dict()
    tuple_to_state = dict()
    
    for i, v in enumerate(yield_states(state_number)):
        state_to_tuple[i] = v
        tuple_to_state[v] = i 

    # Calculate the number of states
    total_states = 1
    for i, _ in enumerate(state_number):
        total_states*=state_number[i]                                                                       
                            
    state_1 = tuple_to_state[corner_points[0][0]]
    state_2 = tuple_to_state[corner_points[0][1]]
    state_3 = tuple_to_state[corner_points[0][2]]
    
    flip = [False, False, False] # Set to True if 
    flip_value = [None,None,None]
    flip_on_graph = [0,0,0]
    
    # Find cuts between corner points.
    for position in np.linspace(0,1,100):
        
        belief_state_x = np.array([0.0 for i in range(number_of_states)])
        belief_state_y = np.array([0.0 for i in range(number_of_states)])
        belief_state_diag = np.array([0.0 for i in range(number_of_states)])
        
        # Should move from bottom left to right over the x-axis
        belief_state_x[state_1] = 1-position
        belief_state_x[state_2] = position

        # Moves from top left to bottom right.
        belief_state_y[state_2] = position
        belief_state_y[state_3] = 1-position

        # Moves from bottom left to top left.
        belief_state_diag[state_3] = 1-position
        belief_state_diag[state_1] = position
        
        epsilon = 0.03
        
        value_x_inspection = belief_state_x @ inspection_alpha
        value_x_continue = float('inf')
        for a in alphas:
            if np.array_equal(a,inspection_alpha):
                continue
            
            if value_x_continue > a @ belief_state_x:
                value_x_continue = a @ belief_state_x
                
        belief_state_0 = np.array([0.0 for i in range(number_of_states)])

        if value_x_continue > value_x_inspection and not  flip[0]:
            flip[0] = True
            flip_on_graph[0] = position-epsilon
            belief_state_0 = np.array([0.0 for i in range(number_of_states)])
            
            belief_state_0[state_1] = 1-flip_on_graph[0]
            belief_state_0[state_2] = flip_on_graph[0]
            
            flip_value[0] = copy.deepcopy(belief_state_0)
        
        value_y_inspection = belief_state_y @ inspection_alpha
        value_y_continue = float('inf')
        for a in alphas:
            if np.array_equal(a,inspection_alpha):
                continue
            
            if value_y_continue > a @ belief_state_y:
                value_y_continue = a @ belief_state_y
                
        if value_y_continue > value_y_inspection and not  flip[1]:
            flip[1] = True
            flip_on_graph[1] = position-epsilon

            belief_state_1 = np.array([0.0 for i in range(number_of_states)])
            
            belief_state_1[state_2] = flip_on_graph[1] 
            belief_state_1[state_3] = 1-flip_on_graph[1] 
            
            flip_value[1] = copy.deepcopy(belief_state_1)

            
        value_diag_inspection = belief_state_diag @ inspection_alpha
        value_diag_continue = float('inf')
        for a in alphas:
            if np.array_equal(a,inspection_alpha):
                continue
            
            if value_diag_continue > a @ belief_state_diag:
                value_diag_continue = a @ belief_state_diag
                
        if value_diag_continue > value_diag_inspection and not  flip[2]:
            flip[2] = True
            flip_on_graph[2] = position - epsilon
            
            belief_state_2 = np.array([0.0 for i in range(number_of_states)])
            
            belief_state_2[state_3] = 1-flip_on_graph[2]
            belief_state_2[state_1] = flip_on_graph[2]
            
            flip_value[0] = copy.deepcopy(belief_state_0)
    
    f.write("Flip values\n")
    f.write("These are used to find the bounds for the 3 region example\n")
    f.write(str(tuple_to_state)+'\n')
    for v in flip_value:
        f.write(str(v))
        f.write('\n')


    # Make a plot showing the optimal policy.
    
    figs, axs = plt.subplots(1,len(corner_points),figsize=(3,3),dpi=300)
    # figs.suptitle("NAME",size = 12, y =0.91) 
    figs.tight_layout(pad=2.2)
    axs.set_aspect('equal', 'box')
    
    plt.setp(axs, xticks=[0,0.5,1], yticks=[0,0.5,1])
    
    # Make a legend

    for v, corners in enumerate(corner_points):
        
        zero_corner = tuple_to_state[corners[0]]
        x_corner = tuple_to_state[corners[1]]
        y_corner = tuple_to_state[corners[2]]
        
        
        # Draw trangles in the bottom right corner (1,1) is the bottom right point.
        
        # Specify the grid on which to draw the data.
        X = np.linspace(0,1,resolution+1)
        Y = np.linspace(0,1,resolution+1)
        X_Mesh, Y_Mesh = np.meshgrid(X, Y)
        Z = 0*(X_Mesh+Y_Mesh)
        
        for x, y in itertools.product(enumerate(X),enumerate(Y)):
            belief_x = x[1] # Label represents axis.
            belief_y = y[1] # Label represents axis.
            i_x = x[0]
            i_y = y[0]
            
            belief_0 = 1-belief_x-belief_y
            if belief_0<0:
                Z[i_y][i_x] = 2
                continue
            
            belief_state = np.array([0.0 for i in range(number_of_states)])
                                
            belief_state[zero_corner] = belief_0
            belief_state[x_corner] = belief_x
            belief_state[y_corner] = belief_y
                        
            optimal_action = 0 # inspect
            
            for a in alphas:
                if optimal_action == 1:
                    break
                
                # If continue is optimal set the optimal action to continue.
                if a @ belief_state < inspection_alpha @ belief_state:
                    optimal_action = 1

            Z[i_y][i_x] = optimal_action
            
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

        colors = ['orange','steelblue',light_grey]
        levels = [-0.5,0.5,1.5,2.5]
                

        axs.set_xlabel("$\pi_{"+str(corners[1])+"}$", size = 11)
        axs.set_ylabel("$\pi_{"+str(corners[2])+"}$", size = 11)

        # axs.text(-0.18,-0.16,r'$\bm{\pi}={}$'.format(corners[0]),size=9)
        axs.text(-0.13,-0.16,r'$\mathbf{s}=$'+str(corners[0]),size=9)
        axs.text(0.91,-0.16,r'$\mathbf{s}=$'+str(corners[1]),size=9)
        axs.text(-0.13,1.05,r'$\mathbf{s}=$'+str(corners[2]),size=9)


        axs.plot([flip_on_graph[0],flip_on_graph[1]],[0,1-flip_on_graph[1]],color = 'black')

        axs.text(flip_on_graph[0]-0.02,-0.095,r'$\bm{\pi}^1$',size=12)
        axs.text(flip_on_graph[1],1-flip_on_graph[1]+0.02,r'$\bm{\pi}^2$',size=12)
        
        CS = axs.contourf(X_Mesh, Y_Mesh, Z, colors = colors, levels = levels)
                            
    # Make a legend
    legend_elements = [Patch(facecolor='steelblue', label='No intervention'),
                        Patch(facecolor='orange', label='Intervention'),
                        Patch(facecolor=light_grey, label='Infeasible')]
    axs.legend(handles=legend_elements, bbox_to_anchor=(1.02 ,1.05), 
              loc='upper left', fontsize=10)

    plt.savefig('00_Exp3_Stage1_plot.png', bbox_inches="tight")
        
    # Detemine optimal stage 2 actions
    
    policy = dict()
    
    for y in yield_states(state_number):
        
        optimal_action = [0]*len(state_number)
        optimal_value = float('inf')
        
        for r in product([0,1], repeat = len(state_number)):
            # Calculate variable cost
            C_v = sum([r[i]*C_replace[i] for i in range(len(state_number))])
            
            # Determine the next state when apply action r to core state y.
            y_next = tuple([(1-r[i])*y[i] for i in range(len(y))])
            s = tuple_to_state[y_next]
            
            new_belief = np.array([0]*total_states)
            new_belief[s] = 1
            
            long_term_cost = float('inf')
            
            for a in alphas:
                if long_term_cost>new_belief @ a:
                    long_term_cost=new_belief @ a
                    
            action_val = C_v + long_term_cost
                    
            action_val = discount_factor*action_val

            if action_val < optimal_value:
                optimal_value = action_val
                optimal_action = r

        policy[y]=optimal_action
        
    state_action_target = []
    for key in policy.keys():
        target = []
        for i, k in enumerate(key):
            target.append(k*(1-policy[key][i]))
        
        state_action_target.append([key,policy[key],target])
        
    for L in state_action_target:
        print("{} -{}-> {}".format(L[0],L[1],L[2]))
        
    f.write("\nStage II: state, action, target\n")
    # Format the stage 2 optimal policy for LaTeX
    for L in state_action_target:
        f.write("{} & {} & {}\\\\\n".format(L[0],L[1],tuple(L[2])))
    

    # Number of degradation states per component.
    state_number = [0]*len(next(iter(Q.keys())))

    for k in Q.keys():
        for i, val in enumerate(k):
            state_number[i] = max(state_number[i],val+1)

    # Create dictionaries relating component based states to linear numbered states.
    state_to_tuple = dict()
    tuple_to_state = dict()
    
    for i, v in enumerate(yield_states(state_number)):
        state_to_tuple[i] = v
        tuple_to_state[v] = i 

    ### Perfect information case ###
    Y = dict()
    
    for x in product(range(3),repeat=2):
        Y[x] = dict()
    
    # Set observation matrix to perfect information.
    for x in product(range(3),repeat=2):
        for y in product(range(3),repeat=2):
            if x == y:
                Y[x][tuple_to_state[y]] = 1
            else:
                Y[x][tuple_to_state[y]] = 0

    f.write("\nPerfect information case\n")
    vector_pruning_modified(Q, Y, C, C_insp, C_replace, discount_factor, accuracy_target=0.3,
                            dmin=0.01)

    ### No information case ###
    for x in product(range(3),repeat=2):
        Y[x] = dict()
        Y[x][0] = 1

    f.write("\nNo information case\n")
    vector_pruning_modified(Q, Y, C, C_insp, C_replace, discount_factor, accuracy_target=0.3,
                            dmin=0.01)

    sim.inspection_alpha = inspection_alpha
    sim.alphas = alphas
    sim.hitting_analysis()
    
    return 0

def main():
        
    f.write("\n\nExperiment 1, 2x2 dgr. & 3 obsv.\n")
    num_exp_1_in_paper()
    
    f.write("\n\nExperiment 3, 3x3 dgr. & 2 obsv.\n")
    num_exp_3_w_plots_in_paper()

    return 0

if __name__ == '__main__':
    main()

    
f.write("\n\nProgram runtime {}".format(time.time()-program_start_time))
f.close()

if time.time() - program_start_time > 60:
    winsound.Beep(frequency, duration)
