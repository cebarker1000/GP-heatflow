from mesh_and_materials.mesh import *
from mesh_and_materials.materials import *
from space.space_and_forms import *
from dirichlet_bc.bc import *
import os
from dolfinx import fem
import pandas as pd
import numpy as np
import gmsh
import yaml
import argparse
import copy
import time
from scipy.spatial import cKDTree
import sys
import contextlib
import ufl
import dolfinx

@contextlib.contextmanager
def suppress_output(enabled):
    if not enabled:
        yield
    else:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                yield

def run_simulation(cfg, mesh_folder, rebuild_mesh=False, visualize_mesh=False, output_folder=None, watcher_points=None, write_xdmf=True, suppress_print=False):
    """
    Run the heatflow simulation with the given configuration.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary loaded from YAML
    mesh_folder : str
        Path to the folder containing mesh.msh and mesh_cfg.yaml
    rebuild_mesh : bool, optional
        Whether to rebuild the mesh and update material tags
    visualize_mesh : bool, optional
        Whether to visualize the mesh and output plots
    output_folder : str, optional
        Where to save XDMF, watcher CSV, and a copy of the used YAML config
    watcher_points : dict or list of dicts, optional
        Points to watch, e.g. {'pside': (z, r), ...}
    write_xdmf : bool, optional
        Whether to write XDMF output
    suppress_print : bool, optional
        If True, suppress all print output
    """
    with suppress_output(suppress_print):


        # -----------------------------------------------------------------------------

        # Start timing
        program_start_time = time.time()
        
        # Create mesh and material boundaries ------------------------------------------
        # Only include insulator, coupler, and sample regions (no diamond or gasket)
        r_sample    = float(cfg['mats']['p_sample']['r'])
        r_ins_oside = float(cfg['mats']['o_ins']['r'])
        r_coupler   = float(cfg['mats']['p_coupler']['r'])
        r_ins_pside = float(cfg['mats']['p_ins']['r'])

        z_ins_oside = float(cfg['mats']['o_ins']['z'])
        z_ins_pside = float(cfg['mats']['p_ins']['z'])
        z_sample    = float(cfg['mats']['p_sample']['z'])
        z_coupler   = float(cfg['mats']['p_coupler']['z'])

        # Mesh boundaries (no diamond or gasket)
        mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler
        mesh_zmax = (z_sample/2) + z_ins_oside + z_coupler
        mesh_rmin = 0.0
        mesh_rmax = r_sample + r_ins_oside

        # build material boundary lists as [zmin, zmax, rmin, rmax]
        bnd_p_ins       = [mesh_zmin, mesh_zmin + z_ins_pside, mesh_rmin, mesh_rmin + r_ins_pside]
        bnd_p_coupler   = [bnd_p_ins[1], bnd_p_ins[1] + z_coupler, mesh_rmin, mesh_rmin + r_coupler]
        bnd_sample      = [bnd_p_coupler[1], bnd_p_coupler[1] + z_sample, mesh_rmin, mesh_rmin + r_sample]
        bnd_o_coupler   = [bnd_sample[1], bnd_sample[1] + z_coupler, mesh_rmin, mesh_rmin + r_coupler]
        bnd_o_ins       = [bnd_o_coupler[1], bnd_o_coupler[1] + z_ins_oside, mesh_rmin, mesh_rmin + r_ins_oside]

        # init materials (no diamond or gasket)
        p_ins       = Material(
                        "p_ins",
                        boundaries=bnd_p_ins,
                        properties={
                            "rho_cv": float(cfg['mats']['p_ins']['rho']) * float(cfg['mats']['p_ins']['cv']),
                            "k": float(cfg['mats']['p_ins']['k'])
                        },
                        mesh_size=float(cfg['mats']['p_ins']['mesh'])
        )
        p_coupler   = Material(
                        "p_coupler",
                        boundaries=bnd_p_coupler,
                        properties={
                            "rho_cv": float(cfg['mats']['p_coupler']['rho']) * float(cfg['mats']['p_coupler']['cv']),
                            "k": float(cfg['mats']['p_coupler']['k'])
                        },
                        mesh_size=float(cfg['mats']['p_coupler']['mesh'])
        )
        p_sample    = Material(
                        "p_sample",
                        boundaries=bnd_sample,
                        properties={
                            "rho_cv": float(cfg['mats']['p_sample']['rho']) * float(cfg['mats']['p_sample']['cv']),
                            "k": float(cfg['mats']['p_sample']['k'])
                        },
                        mesh_size=float(cfg['mats']['p_sample']['mesh'])
        )
        o_coupler   = Material(
                        "o_coupler",
                        boundaries=bnd_o_coupler,
                        properties={
                            "rho_cv": float(cfg['mats']['o_coupler']['rho']) * float(cfg['mats']['o_coupler']['cv']),
                            "k": float(cfg['mats']['o_coupler']['k'])
                        },
                        mesh_size=float(cfg['mats']['o_coupler']['mesh'])
        )
        o_ins       = Material(
                        "o_ins",
                        boundaries=bnd_o_ins,
                        properties={
                            "rho_cv": float(cfg['mats']['o_ins']['rho']) * float(cfg['mats']['o_ins']['cv']),
                            "k": float(cfg['mats']['o_ins']['k'])
                        },
                        mesh_size=float(cfg['mats']['o_ins']['mesh'])
        )
        materials = [p_ins, p_coupler, p_sample, o_coupler, o_ins]

        # build mesh
        gmsh_domain = Mesh(
            name='mesh.msh',
            boundaries=[mesh_zmin, mesh_zmax, mesh_rmin, mesh_rmax],
            materials=materials
        )

        # Determine mesh folder and mesh_cfg path
        mesh_cfg_path = os.path.join(mesh_folder, 'mesh_cfg.yaml')
        mesh_file_path = os.path.join(mesh_folder, 'mesh.msh')

        if rebuild_mesh:
            gmsh_domain.build_mesh()

            # Get material tags
            mat_names = [mat.name for mat in materials]
            mat_tags = [getattr(mat, '_tag', None) for mat in materials]
            # Ensure mat_tags is a list of valid tags (replace None with -1 or similar if needed)
            mat_tags = [tag if tag is not None else -1 for tag in mat_tags]
            mat_tag_map = dict(zip(mat_names, mat_tags))

            # Ensure mesh folder exists
            os.makedirs(mesh_folder, exist_ok=True)

            # Make a deep copy of cfg and add material_tags only to the copy
            mesh_cfg = copy.deepcopy(cfg)
            mesh_cfg['material_tags'] = mat_tag_map

            # Write mesh_cfg to mesh_cfg.yaml in mesh folder
            with open(mesh_cfg_path, 'w') as f:
                yaml.safe_dump(mesh_cfg, f)

            # Write mesh to mesh.msh
            gmsh_domain.write(mesh_file_path)

        else:
            # Error handler: check for mesh.msh and mesh_cfg.yaml
            missing = []
            if not os.path.isfile(mesh_file_path):
                missing.append('mesh.msh')
            if not os.path.isfile(mesh_cfg_path):
                missing.append('mesh_cfg.yaml')
            if missing:
                raise FileNotFoundError(f"Missing required file(s) in {mesh_folder}: {', '.join(missing)}")
            with open(mesh_cfg_path, 'r') as f:
                mesh_cfg = yaml.safe_load(f)

            mat_tag_map = mesh_cfg.get('material_tags', {})

        # visualize mesh
        if visualize_mesh:
            gmsh.initialize()
            gmsh.open(mesh_file_path)
            gmsh.fltk.initialize()
            gmsh.fltk.run()

        # convert gmsh mesh to dolfinx
        from dolfinx.io import gmshio
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(mesh_file_path)
        domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, COMM, 0, 2)
        gmsh.finalize()

        mesh_tag_map = mesh_cfg['material_tags']
        # -----------------------------------------------------------------------------


        # Read experimental heating data  -----------------------------------------------
        # Read heating data from CSV file specified in config
        import pandas as pd
        heating_file = cfg['heating']['file']
        df_heat = pd.read_csv(heating_file)
        
        # Only convert numeric columns to numeric, not the entire DataFrame
        df_heat = (df_heat
                    .sort_values('time')
                    .assign(
                        time=pd.to_numeric(df_heat['time'], errors='coerce'),
                        temp=pd.to_numeric(df_heat['temp'], errors='coerce')
                    )
                    .dropna(subset=['time', 'temp'])  # Only drop rows where time or temp is NaN
                    .reset_index(drop=True))

        # Ensure the CSV has the expected columns
        if 'temp' not in df_heat.columns:
            raise ValueError(f"Heating CSV file {heating_file} must contain a 'temp' column")
        if 'time' not in df_heat.columns:
            raise ValueError(f"Heating CSV file {heating_file} must contain a 'time' column")

        # Normalize temperature data for heating curve
        df_heat['temp normed'] = (df_heat['temp'] - df_heat['temp'].iloc[0]) / (df_heat['temp'].max() - df_heat['temp'].min())
        # -----------------------------------------------------------------------------


        # Initialize function spaces & material properties ----------------------------------------------------
        V = fem.functionspace(domain, ("Lagrange", 1))
        Q = fem.functionspace(domain, ("DG", 0))

        print('Assigning material properties...')


        # assumes each Material object has a .tag and .properties dict
        tag_to_k = {mat_tag_map[mat.name]: mat.properties["k"] for mat in materials}
        tag_to_rho_cv = {mat_tag_map[mat.name]: mat.properties["rho_cv"] for mat in materials}

        cell_tag_array = cell_tags.values

        kappa_per_cell = np.array([tag_to_k[tag] for tag in cell_tag_array])
        rho_cv_per_cell = np.array([tag_to_rho_cv[tag] for tag in cell_tag_array])

        # assign to DG0 functions
        kappa = fem.Function(Q)
        rho_cv = fem.Function(Q)

        kappa.x.array[:] = kappa_per_cell
        rho_cv.x.array[:] = rho_cv_per_cell
        kappa.x.scatter_forward()
        rho_cv.x.scatter_forward()
        print('Material properties assigned.')
        # ------------------------------------------------------------------------------------------


        # Simulation  & heating parameters --------------------------------------------------------
        t_final = float(cfg['timing']['t_final'])
        num_steps = int(cfg['timing']['num_steps'])
        dt = t_final / num_steps
        ic_temp = float(cfg['heating']['ic_temp'])

        heating_FWHM = float(cfg['heating']['fwhm'])
        # ------------------------------------------------------------------------------------------


        # Build variational forms ------------------------------------------------------
        u_n = fem.Function(V)
        u_n.x.array[:] = np.full_like(u_n.x.array, ic_temp) # assign initial temperature
        u_n.x.scatter_forward()

        x = ufl.SpatialCoordinate(domain)
        r = x[1] # y-coord is radial direction

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, PETSc.ScalarType(0))

        lhs = (
            rho_cv * u * v *  r * ufl.dx
            + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) *  r * ufl.dx  
        )
        rhs = (
            rho_cv * u_n * v *  r * ufl.dx
            + dt * f * v * r *  ufl.dx
        )
        lhs_form = fem.form(lhs)
        rhs_form = fem.form(rhs)
        # -----------------------------------------------------------------------------


        # Define boundary conditions ---------------------------------------------------
        # get heating curve
        pside_heating_interp = lambda t: np.interp(t, 
                                                   df_heat['time'], 
                                                   df_heat['temp'], 
                                                   left=df_heat['temp'].iloc[0],
                                                   right=df_heat['temp'].iloc[-1])

        offset = df_heat['temp'].iloc[0] - ic_temp # start from ic temp
        def heating_offset(t):
            return float(pside_heating_interp(t)) - offset

        # gaussian profile for inner boundary
        coeff = -4.0 * np.log(2.0) / heating_FWHM**2
        y_center = 0.0

        def gaussian(x, y, t):
            amp = heating_offset(t)
            return (amp - ic_temp) * np.exp(coeff * (y - y_center)**2) + ic_temp

        obj_bcs = []
        left_bc = RowDirichletBC(V, 'left', value=ic_temp)
        right_bc = RowDirichletBC(V, 'right', value=ic_temp)
        bottom_bc = RowDirichletBC(V, 'top', value=ic_temp)
        inner_bc = RowDirichletBC(
            V,
            'x',
            coord=p_coupler.boundaries[0],
            length=abs(r_sample)*2,
            center=0.0,
            value=gaussian,
        )
        obj_bcs = [left_bc, right_bc, bottom_bc, inner_bc] # custom object
        bcs = [bc.bc for bc in obj_bcs] # dolfinx bc object
        # -----------------------------------------------------------------------------


        # Assemble matrix and vector ---------------------------------------------------
        from dolfinx.fem.petsc import assemble_matrix, create_vector

        A = assemble_matrix(lhs_form, bcs=bcs)
        A.assemble()
        b = create_vector(rhs_form)
        b.assemble()
        # -----------------------------------------------------------------------------


        # Initialize solver ------------------------------------------------------------
        solver = PETSc.KSP().create(A.getComm())
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        # -----------------------------------------------------------------------------


        # Set up output folder
        if output_folder is not None:
            save_folder = output_folder
            os.makedirs(save_folder, exist_ok=True)
            # Save a copy of the used YAML config
            with open(os.path.join(save_folder, 'used_config.yaml'), 'w') as f:
                yaml.safe_dump(cfg, f)
        else:
            save_name = 'refactor_test'
            outputs_folder = os.path.join(os.getcwd(), 'sim_outputs')
            save_folder = os.path.join(outputs_folder, save_name)
            os.makedirs(save_folder, exist_ok=True)

        xdmf_path = os.path.join(save_folder, "output.xdmf")
        watcher_csv_path = os.path.join(save_folder, "watcher_points.csv")

        # Initalize xdmf output --------------------------------------------------------
        from dolfinx import io
        if write_xdmf:
            xdmf = io.XDMFFile(domain.comm, xdmf_path, "w")
            xdmf.write_mesh(domain)
        else:
            xdmf = None

        u_n.name = 'Temperature (K)'
        if write_xdmf:
            xdmf.write_function(u_n, 0.0) # write initial 
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # Pre-compute mesh node coordinates (needed for both watcher points and axis data)
        mesh_coords = domain.geometry.x[:, :2]  # (z, r) for every node

        # Watcher point setup ----------------------------------------------------------
        watcher_data = None
        watcher_names = None
        watcher_time = None
        if watcher_points is not None:
            if isinstance(watcher_points, dict):
                watcher_names = list(watcher_points.keys())
                watcher_coords = list(watcher_points.values())
            elif isinstance(watcher_points, list):
                watcher_names = [pt['name'] for pt in watcher_points]
                watcher_coords = [pt['coords'] for pt in watcher_points]
            else:
                raise ValueError("watcher_points must be a dict or list of dicts")
            watcher_data = {name: [] for name in watcher_names}
            watcher_time = []
            # Nearest-node mapping for watcher points
            tree = cKDTree(mesh_coords)
            watcher_nodes = []
            for coords in watcher_coords:
                _, node_idx = tree.query(coords)
                watcher_nodes.append(node_idx)
        else:
            watcher_names = []
            watcher_coords = []
            watcher_data = {}
            watcher_time = []

        # Radial band nodes (0 < r ≤ 0.1 µm) for radial-gradient CSV --------------------
        band_width = 0.1e-6  # 0.1 micrometre in metres
        band_mask = (mesh_coords[:, 1] > 0.0) & (mesh_coords[:, 1] <= band_width)
        band_nodes = np.where(band_mask)[0]

        # --- DEBUG LINE REQUESTED BY USER ---
        if len(band_nodes) > 0:
            mean_r = mesh_coords[band_nodes, 1].mean()   # radial coord of those nodes
            beta = mean_r / band_width
            print(f"--- Radial Band Analysis ---")
            print(f"  Number of nodes in band: {len(band_nodes)}")
            print(f"  Band width (Δr): {band_width:.2e} m")
            print(f"  Mean radial position of nodes in band: {mean_r:.2e} m")
            print(f"  β = mean_r / Δr = {beta:.4f}")
            if beta > 0.95:
                print(f"  Hypothesis Confirmed: Nodes are clustered near the outer edge (β ≈ 1).")
            elif beta < 0.55 and beta > 0.45:
                 print(f"  Hypothesis Disproved: Nodes are uniformly distributed (β ≈ 0.5).")
            else:
                print(f"  Result: Node distribution is neither fully clustered nor uniform.")
            print(f"--------------------------")
        else:
            print("--- Radial Band Analysis ---")
            print("  No nodes found in the specified radial band.")
            print(f"--------------------------")


        # Build mapping z -> node list within the band
        from collections import defaultdict
        z_to_nodes = defaultdict(list)
        for n in band_nodes:
            z_val = mesh_coords[n, 0]
            z_to_nodes[z_val].append(n)

        # Keep only z positions that actually have nodes
        z_band = np.array(sorted(z_to_nodes.keys()))
        band_node_groups = [z_to_nodes[z] for z in z_band]

        # Containers for gradient over time
        gradient_rows = []  # one list per timestep (averaged grads per z)
        gradient_time = []
        
        # NEW: Containers for raw gradient over time at r=0 nodes
        raw_gradient_rows = []  # one list per timestep (raw grads at r=0 nodes)
        raw_gradient_time = []
        
        print('Setting up radial heat flux sampling...')
        
        # NEW: Find nodes exactly on r=0 for raw gradient calculation
        r_tolerance = 1e-12  # Very small tolerance for r=0
        r_zero_mask = np.abs(mesh_coords[:, 1]) <= r_tolerance
        r_zero_nodes = np.where(r_zero_mask)[0]
        
        # Sort r=0 nodes by z-coordinate for consistent ordering
        r_zero_z_positions = mesh_coords[r_zero_nodes, 0]
        sort_indices = np.argsort(r_zero_z_positions)
        r_zero_nodes = r_zero_nodes[sort_indices]
        r_zero_z_positions = r_zero_z_positions[sort_indices]
        
        print(f"Found {len(r_zero_nodes)} nodes exactly on r=0 axis")
        print(f"r=0 nodes z-range: [{r_zero_z_positions.min():.6e}, {r_zero_z_positions.max():.6e}]")
        
        # Setup computation of radial heat flux (existing method for smoothing)
        V_vec = fem.functionspace(domain, ('Lagrange', 1, (2,)))

        g = ufl.TrialFunction(V_vec)
        w = ufl.TestFunction(V_vec)

        x         = ufl.SpatialCoordinate(domain)
        r_coord   = x[1]

        a_proj_ufl = ufl.inner(g, w) * r_coord * ufl.dx          # weighted mass matrix
        a_proj     = fem.form(a_proj_ufl)                        # wrap!
        A_proj     = assemble_matrix(a_proj)
        A_proj.assemble()

        solver_proj = PETSc.KSP().create(A_proj.getComm())
        solver_proj.setOperators(A_proj)
        solver_proj.setType(PETSc.KSP.Type.PREONLY)
        pc_proj = solver_proj.getPC()
        pc_proj.setType(PETSc.PC.Type.LU)
        pc_proj.setFactorSolverType("mumps")

        grad_smooth = fem.Function(V_vec, name = 'grad_smooth')

        # ---- once, after mesh_coords is defined ----------------------------
        dz_bin = 0.2e-6                  
        z_min, z_max = mesh_coords[:,0].min(), mesh_coords[:,0].max()
        bin_edges = np.arange(z_min, z_max+dz_bin, dz_bin)

        # For every bin, keep indices of nodes that fall inside AND in the radial band
        band_mask = (mesh_coords[:,1] > 0.0) & (mesh_coords[:,1] <= 0.25e-6)
        bin_to_nodes = [[] for _ in range(len(bin_edges)-1)]
        for n in np.where(band_mask)[0]:
            z = mesh_coords[n,0]
            k = np.searchsorted(bin_edges, z) - 1
            if 0 <= k < len(bin_to_nodes):
                bin_to_nodes[k].append(n)

        # Keep bins that actually have nodes
        z_centres  = []
        node_groups = []
        for k, nodes in enumerate(bin_to_nodes):
            if nodes:
                z_centres.append(0.5*(bin_edges[k]+bin_edges[k+1]))
                node_groups.append(nodes)

        #  z_centres = np.array([...])   # columns of the CSV

        # Time stepping loop -----------------------------------------------------------
        from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc
        for x in obj_bcs:
            x.update(0.0)
        progress_interval = max(1, num_steps // 10)

        step_times = []
        loop_start_time = time.time()

        print('Beginning loop...')
        startup_end_time = time.time()
        startup_time = startup_end_time - program_start_time
        for step in range(num_steps):
            step_start = time.time()
            t = (step+1)*dt
            inner_bc.update(t)

            with b.localForm() as local_b:
                local_b.set(0)
            assemble_vector(b, rhs_form)
            apply_lifting(b, [lhs_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)
            solver.solve(b, u_n.x.petsc_vec)
            u_n.x.scatter_forward()

            # Get smoothed gradient (existing method for band-based averaging)
            rhs_proj_ufl = ufl.inner(ufl.grad(u_n), w) * r_coord * ufl.dx
            rhs_proj     = fem.form(rhs_proj_ufl)                    # wrap!
            b_proj       = create_vector(rhs_proj)
            assemble_vector(b_proj, rhs_proj)

            solver_proj.solve(b_proj, grad_smooth.x.petsc_vec)
            grad_smooth.x.scatter_forward()

            # record averaged radial gradient in the band (existing method)
            grad_arr = grad_smooth.x.array.reshape(-1,2)
            radial_vals = [float(np.mean(grad_arr[nodes,1])) for nodes in node_groups]
            gradient_rows.append(radial_vals)
            gradient_time.append(t)

            # NEW: Get raw gradient at r=0 nodes
            raw_grad_vals = []
            for node_idx in r_zero_nodes:
                # Get the gradient at this specific r=0 node
                raw_grad_r = grad_arr[node_idx, 1]  # r-component of gradient
                raw_grad_vals.append(float(raw_grad_r))
            
            raw_gradient_rows.append(raw_grad_vals)
            raw_gradient_time.append(t)

            if write_xdmf:
                xdmf.write_function(u_n, t)
            if watcher_points is not None:
                watcher_time.append(t)
                for name, coords, node_idx in zip(watcher_names, watcher_coords, watcher_nodes):
                    # Get temperature directly from nearest node
                    try:
                        val = u_n.x.array[node_idx]
                    except:
                        val = np.nan
                    watcher_data[name].append(val)

            step_end = time.time()
            step_times.append(step_end - step_start)

            if (step + 1) % progress_interval == 0 or (step + 1) == num_steps:
                percent = int((step + 1) / num_steps * 100)
                # Calculate average time for the current progress interval
                interval_start = max(0, len(step_times) - progress_interval)
                interval_steps = step_times[interval_start:]
                avg_interval = sum(interval_steps) / len(interval_steps)
                print(f"Simulation progress: {percent}% (step {step + 1}/{num_steps}) | Avg time/step (interval): {avg_interval:.4f} s")

        if write_xdmf:
            xdmf.close()

        # Save watcher data to CSV
        if watcher_points is not None:
            import pandas as pd
            df = pd.DataFrame({'time': watcher_time})
            for name in watcher_names:
                df[name] = watcher_data[name]
            df.to_csv(watcher_csv_path, index=False)

        # Write radial gradient CSV only if we have data (existing smoothed method)
        if gradient_rows:
            radial_grad_csv_path = os.path.join(save_folder, "radial_gradient.csv")
            grad_df = pd.DataFrame(gradient_rows, columns=z_centres)
            grad_df.index = gradient_time
            grad_df.index.name = 'time'
            grad_df.to_csv(radial_grad_csv_path)

        # NEW: Write raw radial gradient CSV at r=0 nodes
        if raw_gradient_rows:
            raw_radial_grad_csv_path = os.path.join(save_folder, "radial_gradient_raw.csv")
            raw_grad_df = pd.DataFrame(raw_gradient_rows, columns=r_zero_z_positions)
            raw_grad_df.index = raw_gradient_time
            raw_grad_df.index.name = 'time'
            raw_grad_df.to_csv(raw_radial_grad_csv_path)
            print(f"Saved raw gradient data at r=0 nodes to {raw_radial_grad_csv_path}")

        # Timing outputs before plotting
        program_end_time = time.time()
        total_time = program_end_time - program_start_time
        loop_time = program_end_time - loop_start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0

        print("\n--- Timing Summary ---")
        print(f"Total time: {total_time:.2f} s")
        print(f"Startup time: {startup_time:.2f} s")
        print(f"Loop time: {loop_time:.2f} s")
        print(f"Average time per step: {avg_step_time:.4f} s")
        print("----------------------\n")
        # -----------------------------------------------------------------------------


if __name__ == '__main__':
    # CLI argument parsing --------------------------------------------------------
    parser = argparse.ArgumentParser(description='Heatflow simulation runner')
    parser.add_argument('--config', type=str, default='simulation_template.yaml', help='Path to the folder containing mesh.msh and mesh_cfg.yaml')
    parser.add_argument('--mesh-folder', type=str, default='meshes', help='Path to the folder containing mesh.msh and mesh_cfg.yaml (or where to save them if not provided)')
    parser.add_argument('--rebuild-mesh', action='store_true', help='Rebuild the mesh and update material tags')
    parser.add_argument('--visualize-mesh', action='store_true', help='Visualize the mesh')
    parser.add_argument('--output-folder', type=str, help='Where to save XDMF, watcher CSV, and a copy of the used YAML config')
    parser.add_argument('--watcher-points', type='dict', help='Points to watch, e.g. {"pside": (z, r),zs ...}')
    parser.add_argument('--write-xdmf', action='store_true', help='Whether to write XDMF output')
    parser.add_argument('--suppress-print', action='store_true', help='Suppress all print output')
    args = parser.parse_args()

    # Load simulation configuration from YAML file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    # -----------------------------------------------------------------------------

    # Run the simulation
    run_simulation(cfg, args.mesh_folder, args.rebuild_mesh, args.visualize_mesh, args.output_folder, args.watcher_points, args.write_xdmf, args.suppress_print)
