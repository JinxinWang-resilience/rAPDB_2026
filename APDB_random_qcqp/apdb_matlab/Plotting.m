%------------ Plotting the result ---------------
clear;
load('results_apdb_both_modes.mat');
c1 = [0.000, 0.447, 0.741];   
c2 = [0.850, 0.325, 0.098];   
c3 = [0.929, 0.694, 0.125];  
c4 = [0.494, 0.184, 0.556];  
c5 = [0.466, 0.674, 0.188]; 
c6 = [0.301, 0.745, 0.933];  
c7 = [0.635, 0.078, 0.184]; 
numsim = numel(oracle_egm);
mode_idx = 1;
    % figure(fignum);
figure(1);
subplot(2,2,1);
ax = gca;
ax.FontSize = 16;      
ax.LineWidth = 1.2;        
p1 = Plot_main(oracle_egm,rel_subopt_egm,c4,'--',numsim,[],0);
p2 = Plot_main(oracle_1(mode_idx,:),rel_subopt_1(mode_idx,:),c1,'-',numsim,[],0);
p3 = Plot_main(oracle_1_ada(mode_idx,:),rel_subopt_1_ada(mode_idx,:),[0.466,0.674,0.188],':',numsim,[],0);
p4 = Plot_main(oracle_r1(mode_idx,:),rel_subopt_r1(mode_idx,:),c7,':',numsim,[],0);

p5 = Plot_main(oracle_xy_1(mode_idx,:),rel_subopt_xy_1(mode_idx,:),c1,'--',numsim,[],0);
p6 = Plot_main(oracle_xy_ada1(mode_idx,:),rel_subopt_xy_ada1(mode_idx,:),c5,'-',numsim,[],0);
p7 = Plot_main(oracle_xy_r1(mode_idx,:),rel_subopt_xy_r1(mode_idx,:),c7,'-',numsim,[],0);
grid 'on'
hx = xlabel({'Number of gradient computations'},'Interpreter','latex');
hy = ylabel({'$|f(x^k)-f^\star|/(1+|f^\star|)$'},'Interpreter','latex');
hx.FontSize = 16;    
hy.FontSize = 16;        

lgd = legend([p1 p2 p3 p4 p5 p6 p7],{'EGM','APDB-yx','rAPDB-yx-ada','rAPDB-yx','APDB-xy','rAPDB-xy-ada','rAPDB-xy'});
lgd.FontSize = 11;lgd.Location = 'northeast';

subplot(2,2,2);
p1 = Plot_main(oracle_egm,rel_infeas_err_egm,c4,'--',numsim,[],0);
p2 = Plot_main(oracle_1(mode_idx,:),rel_infeas_1(mode_idx,:),c1,'-',numsim,[],0);
p3 = Plot_main(oracle_1_ada(mode_idx,:),rel_infeas_1_ada(mode_idx,:),[0.466,0.674,0.188],':',numsim,[],0);
p4 = Plot_main(oracle_r1(mode_idx,:),rel_infeas_r1(mode_idx,:),c7,':',numsim,[],0);

p5 = Plot_main(oracle_xy_1(mode_idx,:),rel_infeas_xy_1(mode_idx,:),c1,'--',numsim,[],0);
p6 = Plot_main(oracle_xy_ada1(mode_idx,:),rel_infeas_xy_ada1(mode_idx,:),c5,'-',numsim,[],0);
p7 = Plot_main(oracle_xy_r1(mode_idx,:),rel_infeas_xy_r1(mode_idx,:),c7,'-',numsim,[],0);
grid 'on'
hx = xlabel({'Number of gradient computations'},'Interpreter','latex');
hy = ylabel({'$\frac{1}{m}\sum_{i=1}^m \max\{g_i(x^k),0\}$'},'Interpreter','latex');
hx.FontSize = 16;    
hy.FontSize = 16; 
lgd = legend([p1 p2 p3 p4 p5 p6 p7],{'EGM','APDB-yx','rAPDB-yx-ada','rAPDB-yx','APDB-xy','rAPDB-xy-ada','rAPDB'});
lgd.FontSize = 11;lgd.Location = 'northeast';

mode_idx = 2;
subplot(2,2,3);
ax = gca;
ax.FontSize = 16;      
ax.LineWidth = 1.2;        
% p1 = Plot_main(oracle_egm,rel_subopt_egm,c4,'--',numsim);
p2 = Plot_main(oracle_1(mode_idx,:),rel_subopt_1(mode_idx,:),c1,'-',numsim,[],0);
p3 = Plot_main(oracle_1_ada(mode_idx,:),rel_subopt_1_ada(mode_idx,:),[0.466,0.674,0.188],':',numsim,[],0);
p4 = Plot_main(oracle_r1(mode_idx,:),rel_subopt_r1(mode_idx,:),c7,':',numsim,[],0);

p5 = Plot_main(oracle_xy_1(mode_idx,:),rel_subopt_xy_1(mode_idx,:),c1,'--',numsim,'o',500);
p6 = Plot_main(oracle_xy_ada1(mode_idx,:),rel_subopt_xy_ada1(mode_idx,:),c5,'-',numsim,'^',700);
p7 = Plot_main(oracle_xy_r1(mode_idx,:),rel_subopt_xy_r1(mode_idx,:),c7,'-',numsim,'+',800);
grid 'on'
hx = xlabel({'Number of gradient computations'},'Interpreter','latex');
hy = ylabel({'$|f(x^k)-f^\star|/(1+|f^\star|)$'},'Interpreter','latex');
hx.FontSize = 16;    
hy.FontSize = 16;        

lgd = legend([p2 p3 p4 p5 p6 p7],{'APDB-yx','rAPDB-yx-ada','rAPDB-yx','APDB-xy','rAPDB-xy-ada','rAPDB-xy'});
lgd.FontSize = 11;lgd.Location = 'northeast';

subplot(2,2,4);
% p1 = Plot_main(oracle_egm,rel_infeas_err_egm,c4,'--',numsim);
p2 = Plot_main(oracle_1(mode_idx,:),rel_infeas_1(mode_idx,:),c1,'-',numsim,[],0);
p3 = Plot_main(oracle_1_ada(mode_idx,:),rel_infeas_1_ada(mode_idx,:),[0.466,0.674,0.188],':',numsim,[],0);
p4 = Plot_main(oracle_r1(mode_idx,:),rel_infeas_r1(mode_idx,:),c7,':',numsim,[],0);

p5 = Plot_main(oracle_xy_1(mode_idx,:),rel_infeas_xy_1(mode_idx,:),c1,'--',numsim,'o',500);
p6 = Plot_main(oracle_xy_ada1(mode_idx,:),rel_infeas_xy_ada1(mode_idx,:),c5,'-',numsim,'^',700);
p7 = Plot_main(oracle_xy_r1(mode_idx,:),rel_infeas_xy_r1(mode_idx,:),c7,'-',numsim,'+',800);
grid 'on'
hx = xlabel({'Number of gradient computations'},'Interpreter','latex');
hy = ylabel({'$\frac{1}{m}\sum_{i=1}^m \max\{g_i(x^k),0\}$'},'Interpreter','latex');
hx.FontSize = 16;    
hy.FontSize = 16; 
lgd = legend([p2 p3 p4 p5 p6 p7],{'APDB-yx','rAPDB-yx-ada','rAPDB-yx','APDB-xy','rAPDB-xy-ada','rAPDB'});
lgd.FontSize = 11;lgd.Location = 'northeast';

function y = Plot_main(oracle,rel_subopt,color,linestyle,numsim,marker,gap)
    cell_leng_oracle = cellfun(@length,oracle,'uni',false);
    cell_maxiter_oracle = cellfun(@max,oracle,'uni',false);
    x_axis = linspace(1, max(cell2mat(cell_maxiter_oracle)),max(cell2mat(cell_leng_oracle)))';
    leng_oracle_max = 1:max(cell2mat(cell_leng_oracle));
    for sim=1:numsim
        int_aux = interp1(oracle{1,sim}, rel_subopt{1,sim}, x_axis, 'nearest', 'extrap');
        ind_f(sim,1) = find(int_aux(40:length(int_aux))>0,1)+40;
        int_rel_subopt{1,sim} = int_aux;
    end
    ind_f_min = min(ind_f);
    for sim=1:numsim
        int_rel_subopt{1,sim}(ind_f_min:ind_f(sim,1)) = int_rel_subopt{1,sim}(ind_f(sim,1));
    end
    color_line = color;
    if isempty(marker)
    y = semilogy(x_axis, mean(cell2mat(int_rel_subopt),2), 'color', ...
        color_line, 'LineWidth', 3,'LineStyle', linestyle);
    else
        y = semilogy(x_axis, mean(cell2mat(int_rel_subopt),2), 'Color', color_line, ...
    'LineWidth', 3, ...
    'LineStyle', linestyle, ...
    'Marker', marker, 'MarkerSize', 10,'MarkerIndices', 1:gap:length(x_axis));
    end
    hold 'on'
end
