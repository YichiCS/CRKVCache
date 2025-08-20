
from utils import get_args



if __name__ == "__main__":
    
    # Load args & device
    args = get_args()
    device = f'cuda:{args.device}'
    
    # Load Models
    model_path_dicts = {
        "llama2": "meta-llama/Llama-2-7b-chat-hf", 
        "vicuna": "./models/vicuna/vicuna-7b-v1.3",
        "guanaco": "./models/guanaco/guanaco-7B-HF", 
        "WizardLM": "./models/WizardLM/WizardLM-7B-V1.0",
        "mpt-chat": "./models/mpt/mpt-7b-chat", 
        "mpt-instruct": "./models/mpt/mpt-7b-instruct",
        "falcon": "./models/falcon/falcon-7b-instruct"
    }
    
    model_path = model_path_dicts[args.models]
    template_name = args.model
    
    adv_string_init = open(args.init_prompt_path, 'r').readlines()
    adv_string_init = ''.join(adv_string_init)

    num_steps = args.num_steps
    batch_size = args.batch_size
    num_elites = max(1, int(args.batch_size * args.num_elites))
    crossover = args.crossover
    num_points = args.num_points
    mutation = args.mutation
    API_key = args.API_key
    
    import pdb
    pdb.set_trace()