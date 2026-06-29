import numpy as np
import random

class Hoop():
    def __init__(self, name, attach_to=[]):
        self.name = name
        self.links_to = []
        self.links_from = []
        self.hoop_result = None
        self.parameter = None
        for h in attach_to:
            self.attach_to_hoop(h)

    def attach_to_hoop(self, other_hoop):
        self.links_from.append(other_hoop)
        other_hoop.links_to.append(self)

    def set_parameter(self, parameter):
        self.parameter = parameter

    def modify_parameter(self, amount):
        self.parameter += amount

    def get_parameter(self):
        return self.parameter
        
    def get_output(self):
        return 0

    def reset(self):
        self.hoop_result = None

    def __str__(self):
        return(', '.join([x.name for x in self.links_from]) +
               ' -> ' +
               self.name +
               ' (' + str(self.parameter) +')'
               ' -> ' +
               ', '.join([y.name for y in self.links_to]))


class InputHoop(Hoop):
    def __init__(self, name, attach_to=[]):
        Hoop.__init__(self, name, attach_to)

    def get_output(self):
        return self.parameter

class WeightHoop(Hoop):
    def __init__(self, name, attach_to=[]):
        Hoop.__init__(self, name, attach_to)

    def get_output(self):
        result = 0
        for hoop in self.links_from:
            result += hoop.get_output()
        return result * self.parameter

class SumHoop(Hoop):
    def __init__(self, name, attach_to=[]):
        Hoop.__init__(self, name, attach_to)

    def get_output(self):
        result = 0
        for hoop in self.links_from:
            result += hoop.get_output()
        return result

class ReluHoop(Hoop):
    def __init__(self, name, attach_to=[]):
        Hoop.__init__(self, name, attach_to)

    def get_output(self):
        result = self.parameter
        for hoop in self.links_from:
            result += hoop.get_output()
        if result < 0 :
            result = 0
        return result

class OutputHoop(Hoop):
    def __init__(self, name, attach_to=[]):
        Hoop.__init__(self, name, attach_to)

    def get_output(self):
        result = 0
        for hoop in self.links_from:
            result += hoop.get_output()
        return result


class Network():
    def __init__(self, name):
        self.name = name
        self.hoops = []
        self.input_hoops = []
        self.output_hoops = []
        self.input_values = [0,0]
        self.hoop_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        self.initialise_networks()
        self.set_input_parameters(self.input_values)
        self.set_hoop_parameters(self.hoop_values)
        self.training_data = [('Forward',(-2,4)), ('Forward',(-1,4)), ('Forward',(0,4)), ('Forward',(1,4)), ('Forward',(2,4)),
                              ('Forward',(-2,3)),('Forward',(-1,3)),('Forward',(0,3)),('Forward',(1,3)),('Forward',(2,3)),
                              ('Left',(-2,2)),('Left',(-1,2)),('Forward',(0,2)),('Right',(1,2)),('Right',(2,2)),
                              ('Left',(-2,1)),('Explode',(-1,1)),('Explode',(0,1)),('Explode',(1,1)),('Right',(2,1)),
                              ('Left',(-2,0)),('Explode',(-1,0)),('Explode',(0,0)),('Explode',(1,0)),('Right',(2,0)),
                              ('Left',(-2,-1)),('Left',(-1,-1)),('Left',(0,-1)),('Right',(1,-1)),('Right',(2,-1)),
                              ('Left',(-2,-2)),('Left',(-1,-2)),('Left',(0,-2)),('Right',(1,-2)),('Right',(2,-2)),
                              ('Left',(-2,-3)),('Left',(-1,-3)),('Left',(0,-3)),('Right',(1,-3)),('Right',(2,-3)),
                             ]
                              
        
    
    def initialise_networks(self):
        ix = InputHoop('inputX')
        iy = InputHoop('inputY')
        
        wx1 = WeightHoop('weightX1', attach_to=[ix])
        wx2 = WeightHoop('weightX2', attach_to=[ix])
        wx3 = WeightHoop('weightX3', attach_to=[ix])
        wy1 = WeightHoop('weightY1', attach_to=[iy])
        wy2 = WeightHoop('weightY2', attach_to=[iy])
        wy3 = WeightHoop('weightY3', attach_to=[iy])
        
        h1s = SumHoop('hidden_1_sum', attach_to=[wx1,wy1])
        h2s = SumHoop('hidden_2_sum', attach_to=[wx2,wy2])
        h3s = SumHoop('hidden_3_sum', attach_to=[wx3,wy3])
        
        h1r = ReluHoop('hidden_1_relu', attach_to=[h1s])
        h2r = ReluHoop('hidden_2_relu', attach_to=[h2s])
        h3r = ReluHoop('hidden_3_relu', attach_to=[h3s])
        
        h1w1 = WeightHoop('hidden_1_weight_1', attach_to=[h1r])
        h1w2 = WeightHoop('hidden_1_weight_2', attach_to=[h1r])
        h1w3 = WeightHoop('hidden_1_weight_3', attach_to=[h1r])
        h1w4 = WeightHoop('hidden_1_weight_4', attach_to=[h1r])
        
        h2w1 = WeightHoop('hidden_2_weight_1', attach_to=[h2r])
        h2w2 = WeightHoop('hidden_2_weight_2', attach_to=[h2r])
        h2w3 = WeightHoop('hidden_2_weight_3', attach_to=[h2r])
        h2w4 = WeightHoop('hidden_2_weight_4', attach_to=[h2r])
        
        h3w1 = WeightHoop('hidden_3_weight_1', attach_to=[h3r])
        h3w2 = WeightHoop('hidden_3_weight_2', attach_to=[h3r])
        h3w3 = WeightHoop('hidden_3_weight_3', attach_to=[h3r])
        h3w4 = WeightHoop('hidden_3_weight_4', attach_to=[h3r])
        
        o1 = OutputHoop('Forward', attach_to=[h1w1,h2w1,h3w1]) # Could be generalised later
        o2 = OutputHoop('Left', attach_to=[h1w2,h2w2,h3w2])
        o3 = OutputHoop('Right', attach_to=[h1w3,h2w3,h3w3])
        o4 = OutputHoop('Explode', attach_to=[h1w4,h2w4,h3w4])
        
        self.input_hoops.append(ix)
        self.input_hoops.append(iy)
        
        self.hoops.append(wx1)
        self.hoops.append(wx2)
        self.hoops.append(wx3)
        self.hoops.append(wy1)
        self.hoops.append(wy2)
        self.hoops.append(wy3)
        
        self.hoops.append(h1s)
        self.hoops.append(h2s)
        self.hoops.append(h3s)
        
        self.hoops.append(h1r)
        self.hoops.append(h2r)
        self.hoops.append(h3r)
        
        self.hoops.append(h1w1)
        self.hoops.append(h1w2)
        self.hoops.append(h1w3)
        self.hoops.append(h1w4)
        
        self.hoops.append(h2w1)
        self.hoops.append(h2w2)
        self.hoops.append(h2w3)
        self.hoops.append(h2w4)
        
        self.hoops.append(h3w1)
        self.hoops.append(h3w2)
        self.hoops.append(h3w3)
        self.hoops.append(h3w4)
        
        self.output_hoops.append(o1)
        self.output_hoops.append(o2)
        self.output_hoops.append(o3)
        self.output_hoops.append(o4)

        self.set_hoop_parameters(np.random.randint(-1,6,24))
    
    def set_input_parameters(self, parameters):
        count = 0
        for hoop in self.input_hoops:
            hoop.set_parameter(parameters[count])
            count += 1
            
    def set_hoop_parameters(self, parameters):
        count = 0
        for hoop in self.hoops:
            hoop.set_parameter(parameters[count])
            self.hoop_values[count] = parameters[count]
            count += 1

    def get_hoop_parameters(self):
        parameters = []
        for hoop in self.hoops:
            parameters += [hoop.get_parameter()]
        return parameters

    def get_scores(self, hoops):
        scores = {}
        for hoop in hoops:
            scores[hoop.name] = hoop.get_output()
        return scores     

    def get_outputs(self, input_values):
        self.set_input_parameters(input_values)
        return self.get_scores(self.output_hoops)

    def get_decision(self, input_values):
        scores = self.get_outputs(input_values)
        return list(scores.keys())[np.argmax(list(scores.values()))] 

    def get_rated_decision(self, input_values, desired_outcome):
        scores = self.get_outputs(input_values)
        #print(scores)
        values=list(scores.values())
        #print(values)
        max_indexes = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        #print(max_indexes)
        best_result = list(scores.keys())[max_indexes[0]]
        #print(best_result)
        if (best_result == desired_outcome):
            # return the distance above the next possibility as a positive score
            #print("Best result returning 1")
            # check that there isnt a similar score, and if there is, downweight it.
            if values.count(scores[best_result]) > 1 :
                # There are multiple best values, so reduce the quality to zero
                #print("Returning zero as there are clashes")
                return 0
            else:
                #print("Returning one is this is good")
                return 1 #list(scores.values())[0]-list(scores.values())[1]
            
        else:
            #return the distance bellow the highest point for the desired outcome as a negative value.
            index = list(scores.keys()).index(desired_outcome)
            #print("Not the best result - desired outcome is "+desired_outcome)
            #print(index)
            result = list(scores.values())[index] - list(scores.values())[max_indexes[0]]
            #print(result)
            return result

    def score_hoop_parameters(self, printout=True):
        # score the square of the quantities, i.e. make large numbers more expencive
        score = (np.array(self.get_hoop_parameters())**2).sum()
        # Also increase the score of zeros, so there arnt as many in the overall mix as they will be boring to calculate
        return score + (np.array(self.get_hoop_parameters())==[0]).sum()*25
    
    def evaluate_all_training_data(self, printout=True):
        decision_score = 0
        hoop_score = 0
        correct_decisions = 0
        for outcome, position in self.training_data:
            decision_score += self.get_rated_decision(position, outcome)
            hoop_score -= self.score_hoop_parameters()*0.0001
            if printout:
                decision = self.get_decision(position)
                print("At " + str(position) + " the decision should be " + outcome + ". It is " + decision + ", status " + str(outcome==decision) + ".")
                if decision == outcome:
                    correct_decisions += 1
        score = decision_score + hoop_score
        if printout:
            print(str(correct_decisions) + " out of " + str(len(self.training_data)) + " decisions were correct. Decision score " + str(decision_score) + ", hoop score " + str(hoop_score) + ", total score " + str(score))
        return score

    def mutate_and_accept(self, itterations=1, changes_per_itteration=1, printout=True):
        for i in range(itterations):
            original_hoop_values = self.get_hoop_parameters()
            original_score = self.evaluate_all_training_data(printout=False)
            for j in range(changes_per_itteration):
                #pick a random hoop
                random_hoop = random.choice(self.hoops)
                random_hoop.modify_parameter(random.randint(-2,2))
            new_score = self.evaluate_all_training_data(printout=False)
            if new_score > original_score:
                if printout:
                    print("Acepting new parameters, new score is " + str(new_score))
            else :
                #print("Reverting                old score is " + str(original_score))
                self.set_hoop_parameters(original_hoop_values)

    def simulated_annealing(self, itterations=1, changes_per_itteration=1, heat=1.0, heat_prob=0.1, printout=True):
        original_hoop_values = self.get_hoop_parameters()
        original_score = self.evaluate_all_training_data(printout=False)    
        for i in range(itterations):
            for j in range(changes_per_itteration):
                #pick a random hoop
                random_hoop = random.choice(self.hoops)
                random_hoop.modify_parameter(random.randint(-2,2))
            new_score = self.evaluate_all_training_data(printout=False)
            if new_score > original_score:
                original_hoop_values = self.get_hoop_parameters()
                original_score = new_score 
                if printout:
                    print("Acepting new parameters, new score is " + str(new_score))
            else :
                if new_score > original_score-heat and random.random() > heat_prob:
                    original_hoop_values = self.get_hoop_parameters()
                    original_score = new_score
                    if printout:
                        print("Acepting new heat params, new score is " + str(new_score))
                else:
                    #print("Reverting                old score is " + str(original_score))
                    self.set_hoop_parameters(original_hoop_values)

    def obj_func(self, x):
        self.set_hoop_parameters(np.asarray(x, dtype=np.int16))
        return self.evaluate_all_training_data(printout=False)

        
    def __str__(self): 
        result = ""
        for hoop in self.input_hoops+self.hoops+self.output_hoops:
            result += (hoop.__str__()+'\n')
        return result


def hoopmini(x):
    network = None
    try:
        network = x[0]
    except:
        pass
    if not isinstance(network, Network):
        network = Network('Creeper')
        #print("Making new creeper")
    network.simulated_annealing(itterations=100, changes_per_itteration=8, heat=1.0, heat_prob=0.1, printout=False)
    return (network, network.evaluate_all_training_data(printout=False))

if __name__ == '__main__':
    from multiprocessing import Pool
    epoc_number = 40
    result = [(1,1)] * epoc_number
    best_results = []

    for i in range(1):
    
        with Pool(12) as p:
            result = sorted(p.map(hoopmini, result),key=lambda x: x[1])

        #print(result[-10:])
        print("Iteration "+str(i+1)+" completed")
        best_results += result[-10:]
        result = result[int(-epoc_number/4):] + result[int(-epoc_number/4):] + [(1,1)] * (int(epoc_number/2))

    print("Final result")
    sorted_best = sorted(best_results,key=lambda x: x[1])
    #print(sorted_best)
    print(sorted_best[-1][0])
    sorted_best[-1][0].evaluate_all_training_data()
