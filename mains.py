import sys
import time
import math

from bayesnet import *

# Set to True during debugging, to run an abbreviated set of experiements
# Set to False to run the full set for completing the assignment
short_experiments_only = True

def build_net(filename: str):
    """ File containing the number of variables, each variables name, domain, and CPT, in topological order
    
        NOTE: Variables must be listed in topological order (i.e., variable after all its parents)
        
        File format:
        number of variables
        var1_name var1_value1 var1_value2 ...
        var1_name var1_parent1 var1_parent2 ...
        var1_cpt_val1 var1_parent1_cpt_val1 ... P(var1|parents)
        var1_cpt_val1 var1_parent1_cpt_val2 ... P(var1|parents)
        ...
        var2_name var2_value1 var2_value2 ...
        ...

        Example:
        5
        burglary t f
        burglary
        t 0.001
        f 0.999
        earthquake t f
        earthquake
        t 0.002
        f 0.998
        alarm t f
        alarm burglary earthquake
        t t t 0.95
        f t t 0.05
        t t f 0.94
        f t f 0.06
        t f t 0.29
        f f t 0.71
        t f f 0.001
        f f f 0.999
    """
    with open(filename) as infile:
        nodes: dict[str, BayesianNode] = {}
        topo_order = []
        num_vars = int(infile.readline().strip())

        for _ in range(num_vars):
            tokens = infile.readline().strip().split()
            name = tokens[0]
            values = tokens[1:]
            topo_order.append(name)

            tokens = infile.readline().strip().split()
            assert(tokens[0] == name)
            parents = tokens[1:]

            cpt_size = len(values) * math.prod([ len(nodes[par].values) for par in parents ])
            cpt: dict[tuple, dict[str,float]] = {}
            for _ in range(cpt_size):
                tokens = infile.readline().strip().split()
                node_val = tokens[0]
                parents_vals = tuple(tokens[1:-1])
                prob = float(tokens[-1])

                if parents_vals not in cpt:
                    cpt[parents_vals] = {}

                cpt[parents_vals][node_val] = prob

            nodes[name] = BayesianNode(name, values, parents, cpt)
        
        return BayesianNetwork(nodes, topo_order)



class SamplingResults:
    """ A simple data class to wrap results from a sampling experiment """
    def __init__(self, method: str, num_samples: int, elapsed_time: float, percent_errors: list[float]):
        self.method = method
        self.num_samples = num_samples
        self.elapsed_time = elapsed_time
        self.percent_errors = percent_errors
        
        

def sampling_experiment(net: BayesianNetwork, query_vars: list[str], evidence: dict[str, str],
                         methods: list[str] = ["likelihood_weighting", "gibbs"]) -> list[SamplingResults]:
    """Samples net to generate P(query_vars | evidence) for different methods and numbers of samples

        methods should be some set of ['rejection', 'likelihood_weighting', 'gibbs']
        The default does not include rejection sampling, because it is soooooo ssssllllloooooowwwwwwww

        Each returned SamplingResults represents several repeated trials, using one method and one number of samples
    """
    NUM_ITERATIONS = 25
    MIN_NUM_SAMPLES = 1000
    MAX_NUM_SAMPLES = (2001 if short_experiments_only else 130000) 

    exact_result = net.get_conditional_prob_distribution(query_vars, evidence)

    gibbs = GibbsSampler(net)

    all_results = []
    
    for method in methods:
        num_samples = MIN_NUM_SAMPLES
        while num_samples <= MAX_NUM_SAMPLES:

            percent_errors = []
            # Start the timer
            start_time = time.perf_counter()

            for _ in range(NUM_ITERATIONS):
                approx_result = net.approx_conditional_prob_distribution(method, num_samples, query_vars, evidence, gibbs)

                for query_vals in exact_result:
                    percent_errors.append(100.0 * (exact_result[query_vals] - approx_result[query_vals]) / exact_result[query_vals])

            # End the timer
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time  # units are seconds

            all_results.append(SamplingResults(method, num_samples, elapsed_time, percent_errors))
            
            num_samples *= 2

    return all_results



def plot_sampling_experiment_results(results: list[SamplingResults], title: str) -> None:
    """ Create a plot of the results from sampling_experiment

        Requires:
        import pandas as pd
        import plotly.express as px

        Install those packages using:
        py -m pip install pandas plotly
    """    

    if 'pd' not in sys.modules:
        import pandas as pd
    
    if 'px' not in sys.modules:
        import plotly.express as px

    # Build a pandas DataFrame
    df: pd.DataFrame = None
    for res in results:
        sub_df = pd.DataFrame({'Percent_Error': res.percent_errors})
        sub_df['Num_Samples'] = res.num_samples
        sub_df['Method'] = res.method
        sub_df['Elapsed_Time'] = res.elapsed_time

        if df is not None:
            df = pd.concat([df, sub_df])
        else:
            df = sub_df

    # Convert to string so the plots don't have large gaps in x axis
    df['Num_Samples'] = df['Num_Samples'].apply(lambda x: f'{x:06}')

    # Plot the error data
    px.box(df, x='Num_Samples', y='Percent_Error', color='Method',
           category_orders={'Method': ['rejection', 'likelihood_weighting', 'gibbs']},
           title=title).show()

    # Plot the timing data
    px.bar(df.groupby(['Num_Samples', 'Method'])['Elapsed_Time'].max().reset_index(),
           x='Num_Samples', y='Elapsed_Time', color='Method', barmode='group',
            category_orders={'Method': ['rejection', 'likelihood_weighting', 'gibbs']},
              title=f'Timing: {title}').show()
    

def run_all_sampling_experiments():
    net = build_net('alarm.txt')

    # Common evidence is upstream from query
    query_vars = ['johncalls','marycalls']
    evidence = {'burglary': 'f', 'earthquake': 'f'}
    exp_result = sampling_experiment(net, query_vars, evidence, ["rejection", "likelihood_weighting", "gibbs"])
    plot_sampling_experiment_results(exp_result, 'Common upstream evidence: P(J,M|B=False,E=False)')

    # Rare evidence is upstream from query
    query_vars = ['johncalls','marycalls']
    evidence = {'burglary': 't', 'earthquake': 't'}
    exp_result = sampling_experiment(net, query_vars, evidence)
    plot_sampling_experiment_results(exp_result, 'Rare upstream evidence: P(J,M|B=True,E=True)')

    # Rare evidence is downstream from query
    query_vars = ['burglary', 'earthquake']
    evidence = {'johncalls': 't', 'marycalls': 't'}
    exp_result = sampling_experiment(net, query_vars, evidence)
    plot_sampling_experiment_results(exp_result, 'Rare downstream evidence: P(B,E|J=True,M=True)')

    # Common evidence is downstream from query
    query_vars = ['burglary', 'earthquake']
    evidence = {'johncalls': 'f', 'marycalls': 'f'}
    exp_result = sampling_experiment(net, query_vars, evidence)
    plot_sampling_experiment_results(exp_result, 'Common downstream evidence: P(B,E|J=False,M=False)')





        
if __name__ == '__main__':
    net = build_net('alarm.txt')
    pprint.pprint(net)

    run_all_sampling_experiments()