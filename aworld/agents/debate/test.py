unique_history = []
history = ['a']
if len(history) >= 2:
    for i in range(len(history)):
        # Check if the current element is the same as the next one
        if i == len(history) - 1 or history[i] != history[i+1]:
            # Add the current element to the result list
            unique_history.append(history[i])

print("chat_history: ", unique_history)