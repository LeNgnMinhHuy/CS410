
import numpy as np

def initialize_population( num_individuals, num_variables ):
    """
    Khởi tạo quần thể gồm num_individuals cá thể. Mỗi cá thể có num_parameters biến.
    
    Arguments:
    num_individuals -- Số lượng cá thể
    num_variables -- Số lượng biến
    
    Returns:
    pop -- Ma trận (num_individuals, num_variables ) chứa quần thể mới được khởi tạo ngẫu nhiên.
    """
    
    ### BẮT ĐẦU CODE TỪ ĐÂY ### 
    pop = np.random.randint(2, size=(num_individuals, num_variables))
    
    ### DỪNG CODE TẠI ĐÂY ###
    
    return pop

def onemax( ind ):
    """
    Hàm đánh giá OneMax: Đếm số bit 1 trong chuỗi nhị phân (cá thể ind).
    
    Arguments:
    ind -- Cá thể cần được đánh giá.

    Returns:
    value -- Giá trị của cá thể ind.
    """
    
    ### BẮT ĐẦU CODE TỪ ĐÂY ###   
    value = np.sum(ind)
    
    ### DỪNG CODE TẠI ĐÂY ###
    
    return value

def trap(ind, k = 5):
    start = 0
    l = len(ind)
    trap_value = []
    while start < l:
        fitness = np.sum(ind[start : start + k])
        trap_value.append(k if fitness == k else k - 1 - fitness)
        start += k
    return np.sum(trap_value)

def truncation_selection(pop, pop_fitness, selection_size):
    selected_indices = np.argsort(pop_fitness)[-selection_size:]
    return selected_indices

def crossover_1X( pop ):
    """
    Hàm biến đổi tạo ra các cá thể con.
    
    Arguments:
    pop -- Quần thể hiện tại.

    Returns:
    offspring -- Quần thể chứa các cá thể con được sinh ra.
    """  
    
    ### BẮT ĐẦU CODE TỪ ĐÂY ###
    num_individuals = len(pop)
    num_parameters = len(pop[0])
    indices = np.arange(num_individuals)
    # Đảo ngẫu nhiên thứ tự các cá thể trong quần thể
    np.random.shuffle(indices)
    offspring = []
    
    for i in range(0, num_individuals, 2):
        idx1 = indices[i]
        idx2 = indices[i+1]
        offspring1 = list(pop[idx1])
        offspring2 = list(pop[idx2])
        
        # Cài đặt phép lai đồng nhất One-point crossover. 
        # Không cần cài đặt đột biến mutation.
        start = np.random.randint(low = 0, high = num_parameters) # Chọn vị trí ngẫu nhiên bắt đầu lai ghép 1 điểm
        for idx in range(start, num_parameters):
            temp = offspring2[idx] 
            offspring2[idx] = offspring1[idx]
            offspring1[idx] = temp

        offspring.append(offspring1)
        offspring.append(offspring2)


    ### DỪNG CODE TẠI ĐÂY ###
    
    offspring = np.array(offspring)
    return offspring

def crossover_UX( pop ):
    """
    Hàm biến đổi tạo ra các cá thể con.
    
    Arguments:
    pop -- Quần thể hiện tại.

    Returns:
    offspring -- Quần thể chứa các cá thể con được sinh ra.
    """  
    
    ### BẮT ĐẦU CODE TỪ ĐÂY ### 
    num_individuals = len(pop)
    num_parameters = len(pop[0])
    indices = np.arange(num_individuals)
    
    # Đảo ngẫu nhiên thứ tự các cá thể trong quần thể
    np.random.shuffle(indices)
    offspring = []
    
    for i in range(0, num_individuals, 2):
        idx1 = indices[i]
        idx2 = indices[i+1]
        offspring1 = list(pop[idx1])
        offspring2 = list(pop[idx2])
        
        # Cài đặt phép lai đồng nhất uniform crossover. 
        # Không cần cài đặt đột biến mutation.
        for idx in range(0, num_parameters):
            r = np.random.rand()
            if r < 0.5:
                temp = offspring2[idx] 
                offspring2[idx] = offspring1[idx]
                offspring1[idx] = temp

        offspring.append(offspring1)
        offspring.append(offspring2)


    ### DỪNG CODE TẠI ĐÂY ###
    
    offspring = np.array(offspring)
    return offspring

def tournament_selection(pop, num_individuals, tournament_size = 4, shuffle = False, use_onemax = True):
    if shuffle == True:
        np.random.shuffle(pop)    
    offspring = pop[0, :]
    start = 0
    evaluations = 0
    while start < num_individuals:
        tournament = pop[start:start + tournament_size, :]
        if use_onemax:
            tournament_fitness = np.array([onemax(ind) for ind in tournament])
            evaluations += 1
            selected_indices = truncation_selection(tournament, tournament_fitness, selection_size=1) 
            selection_set = tournament[selected_indices]
            selection_fitness = tournament_fitness[selected_indices]
        else:
            tournament_value = np.array([trap(ind) for ind in tournament])
            evaluations += 1
            selected_indices = truncation_selection(tournament, tournament_value, selection_size=1)
            selection_set = tournament[selected_indices]
            selection_fitness = tournament_value[selected_indices]
            
        offspring = np.vstack([selection_set, offspring])
        
        start += tournament_size
    
    return offspring[: -1, :], evaluations

def convergence(pop):
    return np.isclose(pop, pop[0]).all()

def sGA_POPOP(num_individuals, num_parameters, use_onemax = True, use_1X = True):
    pop = initialize_population(int(num_individuals), num_parameters)
    pop_fitness = np.array([onemax(ind) for ind in pop])
    number_of_evaluations = 1

    selection_size = num_individuals // 2

    while convergence(pop) == False: 
        if use_1X == True:
            offspring = crossover_1X(pop)
        else:
            offspring = crossover_UX(pop)
            
        pop_pool = np.vstack([pop, offspring])
        
        first_tournament, evaluations_1 = tournament_selection(pop_pool, pop_pool.shape[0], use_onemax = use_onemax)
        second_tournament, evaluations_2 = tournament_selection(pop_pool, pop_pool.shape[0], shuffle=True, use_onemax = use_onemax)
        number_of_evaluations += evaluations_1 + evaluations_2 
        pop = np.vstack([first_tournament, second_tournament])
            
        pop_fitness = np.array([onemax(ind) for ind in pop])
        number_of_evaluations += 1
    number_of_evaluations += 1
    return np.max(pop_fitness) == num_parameters, number_of_evaluations # if there is a gen with all value 1, return True. Else return False

"""##BISERTION"""

def find_upper_bound(num_parameters, MSSV, use_onemax=True, use_1X = True):
    n_upper = 4
    success = False
    while success == False:
        n_upper = n_upper * 2
        random_seed = MSSV
        for i in range(0, 10):
            np.random.seed(random_seed)
            success, number_of_evaluations = sGA_POPOP(n_upper, num_parameters, use_onemax=use_onemax, use_1X = use_1X) 
            random_seed += 1
            if success == False:
                break
            
        if n_upper > 8192:
            break
        
    return n_upper

def find_MRPS(num_parameters, n_upper, MSSV, use_onemax=True, use_1X = True):
    n_lower = n_upper // 2
    number_of_evaluations_l = []
    while (n_upper - n_lower)/n_upper > 0.1:
        n = (n_upper + n_lower) / 2
        success = True
        tmp_evaluations = []
        random_seed = MSSV
        for i in range(0, 10):
            np.random.seed(random_seed)
            success, evaluations = sGA_POPOP(n_upper, num_parameters, use_onemax=use_onemax, use_1X = use_1X) 
            tmp_evaluations.append(evaluations)
            random_seed += 1
            if success == False:
                break
            
        if success:
            n_upper = n
            number_of_evaluations_l.extend(tmp_evaluations)
        else: 
            n_lower = n
            
        if (n_upper - n_lower) <= 2:
            break
    
    return n_upper, np.average(number_of_evaluations_l)

def bisection(num_parameters, MSSV, use_onemax=True, use_1X = True):
    MRPS = []
    average_number_of_evaluations = []
    for i in range (0, 10):
        print(f"Bisection: {i + 1}")
        n_upper = find_upper_bound(num_parameters, MSSV, use_onemax=use_onemax, use_1X=use_1X)
        if(n_upper > 8192):
            return [], []
        n_upper, average = find_MRPS(num_parameters, n_upper, MSSV, use_onemax=use_onemax, use_1X = use_1X)
        average_number_of_evaluations.append(average)
        MRPS.append(n_upper)
        MSSV += 10
    return  MRPS, average_number_of_evaluations

l=[10, 20, 40, 80, 160] # problem size

OneMax_1X_result = {'MRPS': [],
                    'average_number_of_evaluations': []}
for i in l:
    MSSV = 20521394
    print("--------------------------------------------------------")
    print(f"Problem size = {i}")
    MRPS, average_number_of_evaluations = bisection(i, MSSV)
    if len(MRPS) == 0:
        print(f"N upper > 8192, Stop at l = {i}")
        break
    OneMax_1X_result['MRPS'].append(MRPS)
    OneMax_1X_result['average_number_of_evaluations'].append(average_number_of_evaluations)

save_path = "OneMax_1X_result.py"
with open(save_path, "w+") as f:
    f.writelines(f"result={OneMax_1X_result}")

OneMax_UX_result = {'MRPS': [],
                    'average_number_of_evaluations': []}
for i in l:
    MSSV = 20521394
    print("--------------------------------------------------------")
    print(f"Problem size = {i}")
    MRPS, average_number_of_evaluations = bisection(i, MSSV, use_1X = False)
    if len(MRPS) == 0:
         print(f"N upper > 8192, Stop at l = {i}")
         break
    OneMax_UX_result['MRPS'].append(MRPS)
    OneMax_UX_result['average_number_of_evaluations'].append(average_number_of_evaluations)

save_path = "OneMax_UX_result.py"
with open(save_path, "w+") as f:
    f.writelines(f"result={OneMax_UX_result}")