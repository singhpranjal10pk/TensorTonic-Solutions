def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code heredef perplexity(prob_distributions, actual_tokens):
    import numpy as np
    log_probs = []
    for i, token in enumerate(actual_tokens):
        prob = prob_distributions[i][token]
        if prob == 0:
            return float('inf')     
        log_probs.append(np.log(prob))
    
    return np.exp(-np.mean(log_probs))