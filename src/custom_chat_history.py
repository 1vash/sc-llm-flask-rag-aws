from collections import deque


class CustomDeque(deque):
    # __str__ is used with print()
    def __str__(self):
        if not self:
            return ""
        else:
            return "".join(map(str, self))


# as we anyway start using OOP and create objects, why not to create a chat object instead of flask sessions?
class CustomChatBufferHistory(object):
    def __init__(self,
                 human_prefix_start='Human:', human_prefix_end='',
                 ai_prefix_start='AI:', ai_prefix_end=''):
        self.memory = CustomDeque()
        # Buffer cleans ony after k+1, `k+1 query invoked firstly and only then popleft happens`.
        self.k = 2

        self.human_prefix_start = human_prefix_start
        self.human_prefix_end = human_prefix_end
        self.ai_prefix_start = ai_prefix_start
        self.ai_prefix_end = ai_prefix_end

    def clean_memory(self):
        self.memory.clear()

    def update_history(self, human_query, ai_response):
        entry = (f'\n{self.human_prefix_start} {human_query} {self.human_prefix_end} '
                 f'\n{self.ai_prefix_start} {ai_response} {self.ai_prefix_end}')

        self.memory.append(entry)
        if len(self.memory) > self.k:
            self.memory.popleft()
