clear
close all

% Written by Arshad Kudrolli
% Poisson equation solved on 2D grid - fluids ak

% Erosion and deposition depending on thresholds selected.

cd('/home/ikhor/physics-erosion/sims/newsim-testgen/');

set = 1;   % 1: Center sticky - binary river; 2: Uniform inlet/river

gridsize = [200 200];   % Size of system to have the grid in 1mm by 1mm (impacts grad values)

%[sx,sy] = meshgrid(1:gridsize(2)/25:gridsize(2),1);  %Initialize streamline plotting values
%[sx,sy] = meshgrid(1:gridsize(2)/14:gridsize(2),1);  %Initialize streamline plotting values

tsteps = 1000; %10000;  %Time step to solve pressure field

istep = 1000;  % Number of iteration steps to review flow fields
dvstep = 16; %16; % Number of time steps to calculate the displacement between flux increments.
deltat = 0.05; % time interval over which particle displacement is integrated. deltat * dvstep matches ramp rate.
phi_max = 0.47; %0.455;  % maximum volume fraction of grains at a given location.

nmax = 3;  % number of simulations examples you want
flow_no_max = 12;  % number of imposed flow starting points you want to get more branched patterns.
iskip = 3; % number of times steps before saving image snapshots. Change to 1 to save every time step
ttstep_max =  253; %201 % Number of time steps for flux increments
K_temp = 0.5;
phi_offset = 0.0;

Q_threshold = 300; % decides local erosion threshold
V_noErode = 1e16;

Q_ramp = 0.4 ;%0.0; %1.5 % 1.0; %0.5 ;%  0.05 ; %0.01; %  ml/min or s
Q_max = 500;

for flow_no=1:flow_no_max   % increment the Q_imposed to get more branched patterns
    for sim_no=1:nmax
        rng(sim_no,'twister'); % Initialize random number generator for reproducible random numbers
        
        %Start Q at finite value to speed up simulations - has significant effect
        %on initial branches
        Q_imposed = 10 + (flow_no-1)*60; % 100
        
        ttstep =  ttstep_max - (flow_no-1)*40;  % Just reduce the number of runs after breakthrough occurs
        
        %For sim5 interface - sim-5
        %Q_imposed = 180 % 250 %170 %200 ; %150 %;120; %ml/min
        
        % Converts from flow rate to flux in experiments
        V_imposed = 10 * Q_imposed/60/29.2/0.185; %  mm/s Initialize
        V_ramp = 10 * Q_ramp/60/29.2/0.185; %increase with ttstep time step
        V_max = 10 * Q_max/60/29.2/0.185;
        V_threshold = 10 * Q_threshold/60/29.2/0.185;  %Threshold for erosion
        V_deposit = 0.9 * V_threshold;  %Threshold for deposition
        
        ismooth = 3; %Number of smoothing steps for conductivity calculations
        
        P = zeros(gridsize);  % Pressure initializ
        for jj=1:gridsize(1)
            P(jj,:) = 1.0 - jj/gridsize(1);
        end
        
        
        B = P;
        %P = zeros(gridsize);  % Pressure
        %B = ones(gridsize);  % dummy for P
        DPX = ones(gridsize);
        DPY = ones(gridsize);
        flux = zeros(tsteps/istep);  %Flux as a function of time, proportional to conductivity
        flux_y = zeros(gridsize(2));
        
        %Conductivity matrix
        K = ones(gridsize);
        DKX = zeros(gridsize);
        DKY = zeros(gridsize);
        
        K_ave = zeros(ttstep);
        ave_flux = zeros(ttstep);
        
        phi = ones(gridsize);
        phi_ave = zeros(ttstep);
        
        %Threshold matrix of the substrate "stickness"
        Thres = ones(gridsize);
        Thres2 = ones(gridsize);
        
        % Porosity initialization
        for i= 1:gridsize(1)
            for j = 1:gridsize(2)
                
                phi(i,j)= phi_max - 0.06 * rand;  % Intialize volume fraction matrix with some randoness
                
                %            Thres(i,j) = V_threshold * (0.5 + (rand+rand+rand+rand)/2.0);  %Intialize substrate stickness matrix
                %            Thres(i,j) = V_threshold * (0.5 + (rand+rand)/1.0);  %Intialize substrate stickness matrix
                Thres(i,j) = V_threshold * (0.5 + (rand)/0.50);  %Intialize substrate stickness matrix
                
                
                if ( j < 10 ),  % Make the left side very sticky to prevent erosion. Approximate the side interface in the experiment
                    Thres(i,j) = V_threshold * V_noErode;
                    phi(i,j) = phi_max;
                end
                if (j - gridsize(2) > -10 ),  % Make the right side very sticky to prevent erosion. Approximate the side interface in the experiment
                    Thres(i,j) = V_threshold * V_noErode;
                    phi(i,j) = phi_max;
                end
                
            end
            
        end
        
        %Main time evolution loop
        for tt = 1:ttstep
            disp(['Flow', int2str(flow_no), ', Sim ', int2str(sim_no), ', step ', int2str(tt)])
            
            % To make the top barrier to have two injection points  - comment out
            % to make it uniform injection
            if set == 1
                for j= 1:gridsize(2)
                    for i= 1:10
                        if (j  > 17) && (j  < gridsize(2) -17),  % Make the center sticky. Approximate the side interface in the experiment
                            Thres(i,j) = V_threshold * V_noErode;
                            phi(i,j) = phi_max;
                        end
                    end
                end
            end
            
            if set == 2
                if (j  > gridsize(2)/2 + 5) || (j  < gridsize(2)/2 -5),  % Make the side sticky. Approximate the side interface in the experiment
                    Thres(i,j) = V_threshold * V_noErode;
                    phi(i,j) = phi_max;
                end
            end
            
            
            % To make the bottom have center drain only
            for j= 1:gridsize(2)
                for i= gridsize(1)-10:gridsize(1)
                    
                    if (j  > gridsize(2)/2+5),  % Make the bottom right very sticky to prevent erosion. Approximate the side interface in the experiment
                        Thres(i,j) = V_threshold * V_noErode;
                        %Make fluid exit through center oriffice
                        phi(i,j) = phi_max;
                    end
                    if (j < gridsize(2)/2-5),  % Make the bottom left very sticky to prevent erosion. Approximate the side interface in the experiment
                        Thres(i,j) = V_threshold * V_noErode;
                        %Make fluid exit through center oriffice
                        phi(i,j) = phi_max;
                    end
                end
            end
            
            
            
            if (V_imposed < V_max),
                V_imposed = V_imposed + V_ramp;
                %      tt_print = tt  %prints time step to screen
                Q_imposed = V_imposed * 60 * 29.2 * 0.185/10; % ml/min
            end
            phi_ave(tt) = sum(sum(phi));
            
            tt_p = mod(tt,iskip);
            if (tt_p == 1) && (tt > 10)
                % write jpg or bmp depending on resolution needed
                filewrite =  strcat('Image-',int2str(flow_no),'-',int2str(sim_no),'-',int2str(tt),'.jpg');
                %      filewrite =  strcat('Image-',int2str(sim_no),'-3-',int2str(tt),'.bmp');
                imwrite(phi,filewrite);
            end
            
            %     Changed the conductivity contrast to be 100 times lower.
            K = (2.0 * 10^(-10) + ((3.0 * 10^(-7)) *(1- phi/phi_max))) * 10^6;    % mm^2  linear map to measured values
            
            %Smooth the conductivity matrix - helps with edge singularities
            for ii = 1:ismooth
                K2 = 0;
                for x=[-1 0 1]
                    for y=[-1 0 1],
                        K2 = K2 + circshift(K, [x y]);
                    end
                end
                K =  K2/9;
                K(1,:) = K(2,:);
                K(gridsize(1),:) = K(gridsize(1)-1,:);
            end
            
            [DKX, DKY] = gradient(K);
            
            %Loop to solve flow field in square grid
            iteration = 0;
            while(iteration < tsteps),
                iteration = iteration + 1;
                % enforce boundary conditions
                P(1,:) = 1;
                P(gridsize(1),:) = 0;
                [DPX,DPY] = gradient(P);
                
                % incrementally solve Poisson's equation by setting every cell equal
                % to the average value of the neighboring cells
                %B = 0;
                B = circshift(P,[1 0]) +circshift(P,[-1 0]) +circshift(P,[0 1]) +circshift(P,[0 -1]);
                P = (B +rdivide(times(DPX,DKX), K) +rdivide(times(DPY, DKY), K))/4;
                
                %Prevent singularities
                P(P>1) = 1;
                P(P<0) = 0;
                
            end
            
            VX = - times(K,DPX);
            VY = - times(K,DPY);
            
            flux_y = sum(VY,2)/gridsize(2);
            
            ave_flux(tt) = sum(flux_y)/gridsize(1);
            K_ave(tt) = ave_flux(tt);
            VX = V_imposed/ave_flux(tt) * VX;   %Scale to imposed flow
            VY = V_imposed/ave_flux(tt) * VY;   %Scale to imposed flow
            
            clims = [0 1];
            % See the image of eroded system over time.
            % Comment following display to speed up
            %         imagesc((phi),clims);   % Review volume fraction
            %imagesc(VX.^2+VY.^2);  % Review flow
            %         imagesc(P,clims);   % Review pressure
            %pause(0.1)   % Comment to speed up
            %     colormap gray
            axis equal tight off
            colorbar;
            
            for i = gridsize(1)-1:-1:2
                for j = gridsize(2)-1:-1:2
                    
                    %Test if particle experience critical shear near edge of low conductivity regions using test condition of K(i,j)
                    if ((sqrt(VX(i,j+1)^2 + VY(i,j+1)^2)) > 2.5 * Thres(i,j) && phi(i,j) > 0.25 && phi(i+1,j+1) < 0.25 && K(i,j) > 0.02),
                        
                        phi_temp0 = phi(i,j);
                        phi(i,j) = phi(i+1,j+1);
                        phi(i+1,j+1) = phi_temp0;
                        
                    end
                    
                    if ((sqrt(VX(i,j-1)^2 + VY(i,j-1)^2)) > 2.5 * Thres(i,j) && phi(i,j) > 0.25 && phi(i+1,j-1) < 0.25 && K(i,j) > 0.02),
                        
                        phi_temp0 = phi(i,j);
                        phi(i,j) = phi(i+1,j-1);
                        phi(i+1,j-1) = phi_temp0;
                        
                    end
                    
                    V_local = (sqrt(VX(i,j)^2 + VY(i,j)^2));
                    
                    % Test if particle position are above critical flow
                    % Two possible models for fluctuations.
                    if (V_local *(1 + 0.2 * (rand -0.5)) > Thres(i,j) && phi(i,j) > 0.25 && K(i,j) > 0.02), % introduce velocity fluctuations at local level
                        %               if (V_local *(1 + 0.0 * (rand -0.5)) > Thres(i,j) && phi(i,j) > 0.25 && K(i,j) > 0.02), %  no fluctuations
                        
                        rand_step = rand;
                        phi_temp = phi(i,j);
                        
                        if (rand_step < 1.0 && phi_temp > phi(i+1,j)),
                            
                            phi(i,j) = phi(i+1,j);
                            phi(i+1,j) = phi_temp;
                            
                            ix = i+1;
                            jy = j;
                            
                            for ttv = 1:dvstep
                                
                                ix_old = ix;
                                jy_old = jy;
                                
                                if (ix > 0 && jy >0 && ix < gridsize(1) && jy < gridsize(2) && jy > 0)
                                    ix = ix + fix(0.5 * ttv * VY(ix_old,jy)*deltat);  %assume grains move 50% of fluid speed
                                    jy = jy + fix(0.5 * ttv * VX(ix_old,jy)*deltat);
                                end
                                
                                if (ix > 0 && jy >0 && ix < gridsize(1) && jy < gridsize(2))
                                    if (phi(ix,jy) < 0.1)
                                        phi_temp2 = phi(ix,jy);
                                        phi(ix,jy) = phi(ix_old,jy_old);
                                        phi(ix_old,jy_old) = phi_temp2;
                                    end
                                else
                                    phi(ix_old,jy_old) = 0.0;
                                    ix = ix_old;
                                    jy = jy_old;
                                end
                                
                                % Particles side ways out of the way if the flow is
                                % non-uniform with small probablity
                                if (ix > 0 && jy >0 && ix < gridsize(1) && jy < gridsize(2)-1 && jy > 1)
                                    if (VY(ix,jy) > 0.9 * VY(ix, jy+1) && phi(ix,jy+1) < 0.1 && rand < 0.1) % instead of 0.2
                                        phi_temp3 = phi(ix,jy+1);
                                        phi(ix,jy+1) = phi(ix,jy);
                                        phi(ix,jy) = phi_temp3;
                                    end
                                    if (VY(ix,jy) > 0.9 * VY(ix, jy-1) && phi(ix,jy-1) < 0.1 && rand < 0.1)
                                        phi_temp3 = phi(ix,jy-1);
                                        phi(ix,jy-1) = phi(ix,jy);
                                        phi(ix,jy) = phi_temp3;
                                    end
                                end
                            end
                        end
                    end
                end
                
            end
            phi(gridsize(1),:) = 0;
            
        end
        
    end
    
end
