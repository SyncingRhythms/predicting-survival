import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# A function to get the id given a row
def get_family_id(row):
	# A dictionary mapping family name to id
	family_id_mapping = {}
	# Find the last name by splitting on a comma
	last_name = row["Name"].split(",")[0]
	# Create the family id
	family_id = "{0}{1}".format(last_name, row["FamilySize"])
	# Look up the id in the mapping
	if family_id not in family_id_mapping:
		if len(family_id_mapping) == 0:
		    current_id = 1
		else:
		    # Get the maximum id from the mapping and add one to it if we don't have an id
		    current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
		family_id_mapping[family_id] = current_id
	return family_id_mapping[family_id]


