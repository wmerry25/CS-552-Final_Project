from torchmetrics.text import ROUGEScore
import string

rouge = ROUGEScore(rouge_keys= "rouge1")

#Output scraper to grab only keywords from input text
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
        word = word.strip(string.punctuation)
        if str.lower(word) in acceptable_terms:
            scraped_output += str.lower(word) + " "
    return scraped_output

#Rouge score eval
def eval(output, ground_truth):
    print(rouge(scrape_output(output),ground_truth))

# eval("""Based on the provided data, I'm looking at the recent trends: Ammonia has increased by 38.46% in the last 10 days to 1.04, Nitrate is steady at 5.7, and Nitrite is steady at 0.85. The strong odor and cloudy water suggest a water quality issue. I suspect a potential Filter Issue or an overload of organic matter. The rising Ammonia is particularly concerning as it's toxic to aquatic life. I recommend checking the filter's performance and considering a water change to mitigate the Ammonia spike.""", "filter issue orp ammonia nitrate nitrite")
# eval("""I'm looking at the recent data: Ammonia has increased by 38.46% in the last 10 days to 1.04, Nitrite is steady at 0.85, and Nitrate is steady at 5.7. The high Ammonia level is likely causing the poor health in your coral and fish, algae bloom, strong odor, and cloudy water. I suspect a Protein Skimmer Failure or Filter Issue as the root cause. High Ammonia is toxic to aquatic life and can lead to further issues. I recommend checking your skimmer and filter for proper function and performing a water change to reduce Ammonia levels.""", "filter issue orp ammonia nitrate nitrite")

# eval("""I'm looking at the recent 10-day data history. I notice a significant 5.05% increase in salinity from 37.63 to 39.63. This rapid change can cause stress to fish and invertebrates, potentially leading to coral closure. The high salinity may be the culprit. I recommend checking the salinity calibration and ATO (Automatic Top-Off) system for potential failure, as it's a potential issue. Verify the salinity reading and adjust it if necessary to prevent further stress. Consider checking the ATO system for proper function to avoid future salinity fluctuations.""","salinity ato failure")
# eval("""I'm looking at the recent data: pH (8.2), Calcium (406.76), Alkalinity (9.71), Magnesium (1291.7), Ammonia (0.24), Nitrite (0.84), Nitrate (5.6), and Temperature (78.1). The salinity has increased by 5.05% over the last 10 days to 39.63, which is a significant change. I suspect a potential issue with the ATO Failure or incorrect salinity adjustments, causing stress to the livestock. The stressed fish and unhealthy invertebrates may be related to this sudden salinity change. I recommend checking the ATO and salinity calibration.""","salinity ato failure")

# eval("""Based on the provided data, I notice a significant 40.24% increase in Alkalinity over 50 days, with a 21.28% increase in the last 10 days, reaching 16.45 dKH. This rapid rise may be causing issues with coral coloration and tissue recession due to excessive alkalinity stress. The hard white buildup is likely calcium carbonate precipitation. I suspect a potential issue with Dosing Pump Failure or over-dosing of alkalinity supplements. Review your dosing regimen to prevent further alkalinity spikes. The data I am looking at includes Alkalinity, Calcium, and pH levels.""", "dosing pump failure alkalinity")
# eval("""I'm examining the recent data: Calcium (406.76), Alkalinity (16.45), Magnesium (1291.7), Phosphate (0.03), Nitrate (5.6), and pH (8.2). The coral issues you're describing could be related to the high alkalinity level (16.45), which has increased by 40.24% in the last 50 days. This may be causing excessive calcium carbonate precipitation, leading to the hard white buildup. I suspect a potential issue with Dosing Pump Failure or over-dosing of alkalinity supplements. I recommend verifying your dosing regimen and checking the pump's functionality.""", "dosing pump failure alkalinity")