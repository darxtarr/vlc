//! VLC command-line interface

use half::f16;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: vlc <command> [options]");
        eprintln!("Commands:");
        eprintln!("  index --emb <file> --d <dim> --m <anchors> --out <path>");
        eprintln!("  info --idx <path>");
        eprintln!("  query --idx <path> --k <k> (test retrieval with synthetic queries)");
        eprintln!("  test  (run with synthetic data)");
        eprintln!("  test-gpu  (run GPU compression test)");
        std::process::exit(1);
    }
    
    match args[1].as_str() {
        "index" => {
            // Parse arguments (manual for now, no clap dependency)
            let mut emb_path = "";
            let mut d = 0;
            let mut m = 0;
            let mut out_path = "";
            
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--emb" => {
                        emb_path = &args[i + 1];
                        i += 2;
                    }
                    "--d" => {
                        d = args[i + 1].parse().expect("Invalid dimension");
                        i += 2;
                    }
                    "--m" => {
                        m = args[i + 1].parse().expect("Invalid anchor count");
                        i += 2;
                    }
                    "--out" => {
                        out_path = &args[i + 1];
                        i += 2;
                    }
                    _ => {
                        eprintln!("Unknown option: {}", args[i]);
                        std::process::exit(1);
                    }
                }
            }
            
            if emb_path.is_empty() || d == 0 || m == 0 || out_path.is_empty() {
                eprintln!("Missing required arguments");
                std::process::exit(1);
            }
            
            println!("Loading embeddings from {}...", emb_path);
            // TODO: Load actual embeddings
            // For now, error out
            eprintln!("Error: Embedding loading not yet implemented");
            eprintln!("Use 'vlc test' to run with synthetic data");
            std::process::exit(1);
        }
        
        "info" => {
            if args.len() < 4 || args[2] != "--idx" {
                eprintln!("Usage: vlc info --idx <path>");
                std::process::exit(1);
            }
            
            let idx_path = &args[3];
            match vlc::io::read_index_info(idx_path) {
                Ok(info) => {
                    println!("Index Information:");
                    println!("  Vectors: {}", info.n_original);
                    println!("  Dimensions: {}", info.d_original);
                    println!("  Compression ratio: {:.2}%", info.compression_ratio * 100.0);
                    println!("  Has Jacobians: {}", info.has_jacobians);
                    println!("  Has residuals: {}", info.has_residuals);
                }
                Err(e) => {
                    eprintln!("Error reading index: {}", e);
                    std::process::exit(1);
                }
            }
        }
        
        "test" => {
            println!("Running test with synthetic data...");

            // Parse optional size argument
            let (n, d, m) = if args.len() > 2 && args[2] == "--large" {
                (10000, 128, 256) // Large test
            } else {
                (300, 64, 10) // Default small test
            };
            
            println!("Generating {} points in {}D with {} anchors", n, d, m);
            
            // Create three cluster centers
            let centers = vec![
                vec![0.0f32; d],  // Center at origin
                vec![5.0f32; d],  // Center at (5,5,5,...)
                vec![10.0f32; d], // Center at (10,10,10,...)
            ];
            
            // Generate points around centers
            let mut points = Vec::with_capacity(n * d);
            for i in 0..n {
                let center_idx = i % 3;
                let center = &centers[center_idx];
                
                for j in 0..d {
                    // Add small noise around center
                    let noise = ((i * d + j) as f32 * 0.1).sin() * 0.5;
                    let val = center[j] + noise;
                    points.push(f16::from_f32(val));
                }
            }
            
            println!("Running compression...");
            
            let config = vlc::AnnealingConfig {
                m,
                initial_temp: 1.0,
                cooling_rate: 0.05,
                learning_rate: 0.2,
                trim_percent: 0.1,
                max_iterations: 50,
                energy_tolerance: 1e-3,
                min_assignment_changes: 1,
                maintenance_interval: 20,
            };
            
            let compressed = vlc::compress(&points, n, d, config);
            
            println!("\nCompression complete!");
            println!("  Iterations: {}", compressed.metadata.iterations);
            println!("  Final energy: {:.4}", compressed.metadata.final_energy);
            println!("  Compression ratio: {:.2}%", compressed.metadata.compression_ratio * 100.0);
            
            // Save to disk
            let out_path = "./test_vlc";
            println!("\nSaving to {}...", out_path);
            match vlc::write_index(&compressed, out_path) {
                Ok(_) => println!("Index saved successfully"),
                Err(e) => eprintln!("Error saving index: {}", e),
            }
            
            // Try to read it back
            println!("\nVerifying saved index...");
            match vlc::read_index(out_path) {
                Ok(loaded) => {
                    println!("Index loaded successfully");
                    println!("  Anchors: {}", loaded.anchor_set.m);
                    println!("  Points: {}", loaded.assignments.n);
                    
                    // Check assignment distribution
                    let counts = loaded.assignments.count_per_anchor(loaded.anchor_set.m);
                    let min_count = *counts.iter().min().unwrap_or(&0);
                    let max_count = *counts.iter().max().unwrap_or(&0);
                    let avg_count = counts.iter().sum::<usize>() as f32 / counts.len() as f32;
                    
                    println!("\nAssignment distribution:");
                    println!("  Min points per anchor: {}", min_count);
                    println!("  Max points per anchor: {}", max_count);
                    println!("  Avg points per anchor: {:.1}", avg_count);
                }
                Err(e) => eprintln!("Error loading index: {}", e),
            }
        }

        "test-gpu" => {
            println!("Running GPU compression test...");

            // Parse optional size argument
            let (n, d, m) = if args.len() > 2 && args[2] == "--large" {
                (10000, 128, 256) // Large test
            } else {
                (1000, 64, 10) // Default test
            };

            println!("Generating {} points in {}D with {} anchors", n, d, m);

            // Create three cluster centers
            let centers = vec![
                vec![0.0f32; d],  // Center at origin
                vec![5.0f32; d],  // Center at (5,5,5,...)
                vec![10.0f32; d], // Center at (10,10,10,...)
            ];

            // Generate points around centers
            let mut points = Vec::with_capacity(n * d);
            for i in 0..n {
                let center_idx = i % 3;
                let center = &centers[center_idx];

                for j in 0..d {
                    // Add small noise around center
                    let noise = ((i * d + j) as f32 * 0.1).sin() * 0.5;
                    let val = center[j] + noise;
                    points.push(f16::from_f32(val));
                }
            }

            let config = vlc::AnnealingConfig {
                m,
                initial_temp: 1.0,
                cooling_rate: 0.05,
                learning_rate: 0.2,
                trim_percent: 0.1,
                max_iterations: 50,
                energy_tolerance: 1e-3,
                min_assignment_changes: 1,
                maintenance_interval: 20,
            };

            // Time GPU
            println!("\nRunning GPU compression...");
            let start = std::time::Instant::now();
            let gpu_result = pollster::block_on(vlc::compress_gpu(&points, n, d, config.clone()))
                .unwrap();
            let gpu_time = start.elapsed();

            println!("\nGPU Results:");
            println!("  Time: {:?}", gpu_time);
            println!("  Iterations: {}", gpu_result.metadata.iterations);
            println!("  Final energy: {:.4}", gpu_result.metadata.final_energy);
            println!("  Compression ratio: {:.2}%", gpu_result.metadata.compression_ratio * 100.0);

            // Time CPU for comparison
            println!("\nRunning CPU compression for comparison...");
            let start = std::time::Instant::now();
            let cpu_result = vlc::compress(&points, n, d, config);
            let cpu_time = start.elapsed();

            println!("\nCPU Results:");
            println!("  Time: {:?}", cpu_time);
            println!("  Iterations: {}", cpu_result.metadata.iterations);
            println!("  Final energy: {:.4}", cpu_result.metadata.final_energy);
            println!("  Compression ratio: {:.2}%", cpu_result.metadata.compression_ratio * 100.0);

            // Compare
            println!("\nComparison:");
            println!("  Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
            let energy_diff = (gpu_result.metadata.final_energy - cpu_result.metadata.final_energy).abs();
            println!("  Energy difference: {:.6} ({:.2}%)",
                energy_diff,
                energy_diff / cpu_result.metadata.final_energy * 100.0);
        }

        "query" => {
            println!("Testing retrieval with synthetic queries...");

            // First, compress synthetic data to get an index
            println!("\n1. Creating compressed index...");
            let n = 1000;
            let d = 64;
            let m = 20;

            // Generate synthetic data (3 clusters)
            let mut points = Vec::with_capacity(n * d);
            let cluster_centers = vec![
                (0.0f32, 0.0f32),
                (5.0f32, 5.0f32),
                (10.0f32, 0.0f32),
            ];

            for point_idx in 0..n {
                let cluster = point_idx % 3;
                let (cx, cy) = cluster_centers[cluster];

                for dim in 0..d {
                    let noise = (point_idx as f32 * 0.01 + dim as f32 * 0.001).sin() * 0.5;
                    let val = if dim < 2 {
                        match dim {
                            0 => cx + noise,
                            1 => cy + noise,
                            _ => unreachable!(),
                        }
                    } else {
                        noise
                    };
                    points.push(f16::from_f32(val));
                }
            }

            // Compress
            let config = vlc::AnnealingConfig {
                m,
                initial_temp: 1.0,
                cooling_rate: 0.01,
                learning_rate: 0.1,
                trim_percent: 0.1,
                max_iterations: 100,
                energy_tolerance: 1e-4,
                min_assignment_changes: 5,
                maintenance_interval: 20,
            };

            let index = vlc::compress(&points, n, d, config);
            println!("   Compressed: {} anchors, {:.2}% ratio",
                     index.anchor_set.m, index.metadata.compression_ratio * 100.0);

            // 2. Generate test queries (from each cluster)
            println!("\n2. Generating test queries...");
            let num_queries = 10;
            let mut queries = Vec::with_capacity(num_queries * d);

            for i in 0..num_queries {
                let cluster = i % 3;
                let (cx, cy) = cluster_centers[cluster];

                for dim in 0..d {
                    let noise = (i as f32 * 0.02 + dim as f32 * 0.002).cos() * 0.3;
                    let val = if dim < 2 {
                        match dim {
                            0 => cx + noise,
                            1 => cy + noise,
                            _ => unreachable!(),
                        }
                    } else {
                        noise
                    };
                    queries.push(f16::from_f32(val));
                }
            }

            // 3. Perform queries
            println!("\n3. Running queries...");
            let k = 10;
            let start = std::time::Instant::now();

            for i in 0..num_queries {
                let query_start = i * d;
                let query = &queries[query_start..query_start + d];
                let results = index.query(query, k, None);

                println!("   Query {}: found {} neighbors", i + 1, results.len());
                if !results.is_empty() {
                    println!("      Top result: point {} at distance {:.4}",
                             results[0].0, results[0].1.sqrt());
                }
            }

            let elapsed = start.elapsed();
            println!("\n4. Performance:");
            println!("   Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
            println!("   Per query: {:.2}ms", elapsed.as_secs_f64() * 1000.0 / num_queries as f64);
            println!("   Queries/sec: {:.0}", num_queries as f64 / elapsed.as_secs_f64());
        }

        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}
