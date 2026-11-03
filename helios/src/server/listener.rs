/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use axum::serve::Listener;
use std::{io, net::SocketAddr};
use tokio::net::{TcpListener, TcpStream};

pub(super) struct HttpListener(TcpListener);

impl HttpListener {
    pub(super) fn new(listener: TcpListener) -> Self {
        Self(listener)
    }
}

impl Listener for HttpListener {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, peer) = Listener::accept(&mut self.0).await;
            match stream.set_nodelay(true) {
                Ok(()) => return (stream, peer),
                Err(error) => {
                    eprintln!(
                        "rejecting HTTP connection from {peer}: cannot set TCP_NODELAY: {error}"
                    );
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.0.local_addr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn every_accepted_socket_disables_nagle() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = tcp.local_addr().unwrap();
            let mut listener = HttpListener::new(tcp);
            assert_eq!(listener.local_addr().unwrap(), address);
            for _ in 0..2 {
                let client = TcpStream::connect(address).await.unwrap();
                client.set_nodelay(false).unwrap();
                let (accepted, peer) = listener.accept().await;
                assert_eq!(peer, client.local_addr().unwrap());
                assert!(accepted.nodelay().unwrap());
                assert!(!client.nodelay().unwrap());
            }
        })
        .await
        .expect("loopback socket acceptance timed out");
    }
}
