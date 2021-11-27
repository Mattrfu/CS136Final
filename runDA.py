import copy

def runStudentDA(context, desires):
    # from example 12.5 in textbook
    num_agents = 3

    print("Proposing preferences: ", desires[:num_agents])
    print("Accepting preferences: ", desires[num_agents:])
    print()

    student_preferences = copy.deepcopy(desires[:num_agents])
    teacher_preferences = desires[num_agents:]
    
    student_partners = [None] * num_agents
    teacher_partners = [None] * num_agents
    
    counter = 1
    while None in student_partners:

        # Step 1: All student ask their top preference remaining beta
        for student in range(0, num_agents):
            if student_partners[student] == None: # student is unassigned a partner
                top_preference = student_preferences[student].pop(0)

                if top_preference == None:
                    # Student prefers themselves
                    student_partners[student] == -1
                    break

                # Step 2: All beta's accept their top preferred suitor, rejecting existing suitor
                beta = teacher_preferences[top_preference]
                # print("Student " + str(student)  + " proposes to Teacher " + str(top_preference))

                # if beta accepts alpha's proposal
                if teacher_partners[top_preference] is None or (
                    # check that the suitor is strictly preferred to the existing partner
                    teacher_preferences[top_preference].index(student) <
                    teacher_preferences[top_preference].index(teacher_partners[top_preference])
                ):

                    # # if beta has a partner
                    if teacher_partners[top_preference]:
                        # this existing alpha partner is now an ex
                        # print('Teacher ' + str(top_preference) + ' rejects  Student ' + str(teacher_partners[top_preference]))
                        # this alpha person has no partner now :(
                        student_partners[teacher_partners[top_preference]] = None
            
                    # log the match
                    # print('Teacher ' + str(top_preference) + ' accepts  Student ' + str(student))
                    student_partners[student] = top_preference
                    teacher_partners[top_preference] = student
                # else:
                    # print('Teacher ' + str(top_preference) + ' rejects  Student ' + str(student))
                    # move on to the next unmatched student
        
        print("Round " + str(counter) + " results -- (student, teacher)")
        counter += 1
        for i in range(0, len(student_partners)):
            print("(" + str(i) + "," + str(student_partners[i]) + ")")

    print()
    print("Everyone is matched. This is a stable matching")

    # return score of Student 1's matching
    student_1_matched_teacher = student_partners[0]
    if student_1_matched_teacher == -1: # matched themselves
        return 0

    return num_agents - desires[:num_agents][0].index(student_1_matched_teacher)

desires = [
        [1, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [0, 2, 1],
        [2, 0, 1],
        [0, 2, 1] 
    ]

context = [1, 0, 2]
student_1_score = runStudentDA(context, desires)

print("Student 0's score is", student_1_score)