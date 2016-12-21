figure('units','normalized','outerposition',[0 0 1 1])
for i= 1:length(doubletap1)
    sensor1 = doubletap1(1:i,1)
    sensor2 = doubletap1(1:i,2)
    sensor3 = doubletap1(1:i,3)
    sensor4 = doubletap1(1:i,4)
    sensor5 = doubletap1(1:i,5)
    sensor6 = doubletap1(1:i,6)
    sensor7 = doubletap1(1:i,7)
    sensor8 = doubletap1(1:i,8)
    
    t = 0:dt:i*dt;
    t = t(1:end-1);
    plot(t,sensor1,'*-','Linewidth',2)
    xlabel('Time in Seconds')
    ylabel('Raw Sensor Value')
    %text(0,0,'Doubletap','Fontsize',20)
    hold on
     plot(t,sensor2,'*-','Linewidth',2)
     plot(t,sensor3,'*-','Linewidth',2)
     plot(t,sensor4,'*-','Linewidth',2)
     plot(t,sensor5,'*-','Linewidth',2)
     plot(t,sensor6,'*-','Linewidth',2)
     plot(t,sensor7,'*-','Linewidth',2)
     plot(t,sensor8,'*-','Linewidth',2)
    legend('Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7','Sensor 8','Location','Best')
    pause(0.001)
    
end

text(0,60,'Double Tap','FontSize',15)