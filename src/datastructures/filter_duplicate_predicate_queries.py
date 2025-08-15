def filter_duplicate_subject_predicate_combinations(query):
    subject_predicate_count = {}
    for pattern in query["triple_patterns"]:
        pattern_split = pattern.split(" ")
        key = (pattern_split[0], pattern_split[1])
        subject_predicate_count[key] = subject_predicate_count.get(key, 0) + 1
    if any(count >= 2 for count in subject_predicate_count.values()):
        return None
    print("Returning!")
    return query
