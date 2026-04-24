def get_model(model_name, args):
    name = model_name.lower()
    if name == "aper_adapter":
        from models.aper_adapter import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == 'ranpac':
        from models.ranpac import Learner
    elif name == "ease":
        from models.ease import Learner
    elif name == 'cofima':
        from models.cofima import Learner
    elif name == 'cprompt':
        from models.cprompt import Learner
    elif name == 'slot':
        from models.slot_learner import Learner
    else:
        assert 0, f"Un-recognized learner name: {name}"
    
    if 'use_slot' in args.keys() and args['use_slot']: 
        from models.slot_learner import LearnerWrapper
        return LearnerWrapper(Learner(args))
    
    return Learner(args)