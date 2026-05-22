function [z, dual_star] = proj_pos_plane(x,y)
flag = 0;
n = length(x);
index_plus = find(y==1);
n1 = length(index_plus);
index_minus = find(y==-1);
n2 = length(index_minus);
x_plus = x(index_plus);
x_minus = x(index_minus);
x_sort = sort([x_plus;-x_minus]);
x0 = 0;
g0 = sum(max(x_plus-x0*ones(n1,1),zeros(n1,1))) - sum(max(x_minus+x0*ones(n2,1),zeros(n2,1)));
g_old = g0;
if g_old == 0
    dual_star = 0;
    flag = 1;
elseif g_old>0
    ind = find(x_sort>0);
    xpp = x_sort(ind);
    x_old = x0;
    for i=1:length(xpp)
       x_new = xpp(i);
       g = sum(max(x_plus-x_new*ones(n1,1),zeros(n1,1))) - sum(max(x_minus+x_new*ones(n2,1),zeros(n2,1)));
       if g_old>=0 && g<=0
           if g_old == g
               dual_star = x_old;
               flag = 1;
               break;
           else
               dual_star = x_old-g_old*(x_new-x_old)/(g-g_old);
               flag = 1;
               break;
           end
       end
       x_old = x_new;
       g_old = g;
    end
    if flag == 0
        x_new = x_old+1;
        g = sum(max(x_plus-x_new*ones(n1,1),zeros(n1,1))) - sum(max(x_minus+x_new*ones(n2,1),zeros(n2,1)));
        dual_star = x_old-g_old*(x_new-x_old)/(g-g_old);
        flag = 1;
    end
else
    ind = find(x_sort<0);
    xmm = x_sort(ind);
    x_old = x0;
    for i=length(xmm):-1:1
       x_new = xmm(i);
       g = sum(max(x_plus-x_new*ones(n1,1),zeros(n1,1))) - sum(max(x_minus+x_new*ones(n2,1),zeros(n2,1)));
       if g_old<=0 && g>=0
           if g_old == g
               dual_star = x_old;
               flag = 1;
               break;
           else
               dual_star = x_old-g_old*(x_new-x_old)/(g-g_old);
               flag = 1;
               break;
           end
       end
       x_old = x_new;
       g_old = g;
    end
    if flag == 0
        x_new = x_old-1;
        g = sum(max(x_plus-x_new*ones(n1,1),zeros(n1,1))) - sum(max(x_minus+x_new*ones(n2,1),zeros(n2,1)));
        dual_star = x_old-g_old*(x_new-x_old)/(g-g_old);
        flag = 1;
    end
end
z = [max(x_plus-dual_star*ones(n1,1),zeros(n1,1)) ; max(x_minus+dual_star*ones(n2,1),zeros(n2,1))];
aux = [z,[index_plus;index_minus]];
aux = sortrows(aux,2);
z = aux(:,1);
end