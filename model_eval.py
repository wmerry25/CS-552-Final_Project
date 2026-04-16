from torchmetrics.text import ROUGEScore

rouge = ROUGEScore(rouge_keys= "rouge1")

def scrape_output(txt):
    acceptable_terms = {
        "ato", "failure", "heater", "dosing", "pump", "filter", "issue", 
        "flow", "livestock", "death", "refugium", "light", "protein", 
        "skimmer", "calcium", "alkalinity", "magnesium", "phosphate", 
        "nitrate", "nitrite", "ammonia", "salinity", "temperature", "ph", "orp", "rising", "increase", "decrease"
    }
    scraped_output = ""
    words = txt.split() if isinstance(txt, str) else txt
    for word in words:
        if str.lower(word) in acceptable_terms:
            scraped_output += str.lower(word) + " "
    return scraped_output

def eval(output, ground_truth, metrics):
    rouge(scrape_output(output),ground_truth)
    print(rouge)
    print(metrics)