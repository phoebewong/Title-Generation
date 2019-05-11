## Evaluation
def rouge_evaluation(prediction_list, ref_list, method = "rouge-1"):
    '''
    Use rouge to evaluate the title similarity

    Parameters:
    ===========
    prediction_list: a list of words in the predicted title (from getPrediction())
    ref_list: a list of words in the actual title (from testY)
    method: method to evaluate, options include: rouge-1, rouge-2 and rouge-l

    Example
    ====
    i = 40
    check_pred = getPrediction(mod, check[i]], idx2word, 250, 20)
    true_title = [idx2word[str(m)] for m in testY[i]
    1gram_f, 1gram_p, 1gram_r = rouge_evaluation(check_pred, true_title)
    '''
    hypothesis = ' '.join(prediction_list)
    reference = ' '.join(ref_list)

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    f, p, r = scores[0][method].values()
    return f, p, r
